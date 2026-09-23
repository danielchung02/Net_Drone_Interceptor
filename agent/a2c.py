"""Advantage Actor-Critic (A2C) with correct timeout bootstrap handling."""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from agent.common import (
    EVAL_METRIC_FIELDS,
    TRAIN_METRIC_FIELDS,
    append_csv_row,
    evaluate_episodes,
    set_seed,
)
from config import ExperimentConfig
from interception_env import InterceptionEnv


class AgentHyperParameters:
    def __init__(self):
        self.hidden_dim = 128
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.learning_rate = 3e-4
        self.rollout_steps = 1_024
        self.value_coef = 0.5
        self.entropy_coef = 1e-3
        self.min_log_std = -5.0
        self.max_log_std = 1.0
        self.max_grad_norm = 0.5

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


def squashed_log_prob(distribution: Normal, raw_action: torch.Tensor) -> torch.Tensor:
    log_tanh_jacobian = 2.0 * (
        math.log(2.0) - raw_action - F.softplus(-2.0 * raw_action)
    )
    return distribution.log_prob(raw_action) - log_tanh_jacobian


def sample_squashed_normal(mean: torch.Tensor, log_std: torch.Tensor, action_masks: torch.Tensor):
    distribution = Normal(mean, torch.exp(log_std))
    raw_action = distribution.rsample()
    action = torch.tanh(raw_action)
    per_dimension = squashed_log_prob(distribution, raw_action)
    log_prob = (per_dimension * action_masks).sum(dim=-1)
    entropy = (distribution.entropy() * action_masks).sum(dim=-1)
    return action, raw_action, log_prob, entropy


def evaluate_squashed_normal(
    mean: torch.Tensor, log_std: torch.Tensor, raw_action: torch.Tensor, action_masks: torch.Tensor
):
    distribution = Normal(mean, torch.exp(log_std))
    action = torch.tanh(raw_action)
    per_dimension = squashed_log_prob(distribution, raw_action)
    log_prob = (per_dimension * action_masks).sum(dim=-1)
    entropy = (distribution.entropy() * action_masks).sum(dim=-1)
    return action, log_prob, entropy


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int,
        min_log_std: float = -5.0,
        max_log_std: float = 1.0,
    ):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, states: torch.Tensor):
        mean = self.actor_mean(states)
        log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std).expand_as(mean)
        values = self.critic(states).squeeze(dim=-1)
        return mean, log_std, values

    def clamp_log_std(self) -> None:
        with torch.no_grad():
            self.log_std.clamp_(self.min_log_std, self.max_log_std)

    def sample_action(self, states: torch.Tensor, action_masks: torch.Tensor):
        mean, log_std, values = self.forward(states)
        action, raw_action, log_prob, entropy_estimate = sample_squashed_normal(mean, log_std, action_masks)
        return action, raw_action, log_prob, entropy_estimate, values

    def evaluate_actions(self, states: torch.Tensor, raw_actions: torch.Tensor, action_masks: torch.Tensor):
        mean, log_std, values = self.forward(states)
        _, log_prob, entropy_estimate = evaluate_squashed_normal(mean, log_std, raw_actions, action_masks)
        return log_prob, entropy_estimate, values

    def deterministic_action(self, states: torch.Tensor) -> torch.Tensor:
        mean, _, _ = self.forward(states)
        return torch.tanh(mean)


def collect_rollout(
    env,
    model: ActorCritic,
    state: np.ndarray,
    num_steps: int,
    device: torch.device,
    running_episode_return: float,
    running_episode_length: int,
):
    """Collect one A2C rollout, preserving true final observations.

    A reset happens only *after* ``next_state`` has been stored.  This is the
    critical detail that lets a timeout use V(final_next_state) rather than
    V(reset_state).
    """

    state_list: List[np.ndarray] = []
    next_state_list: List[np.ndarray] = []
    raw_action_list: List[np.ndarray] = []
    action_mask_list: List[np.ndarray] = []
    reward_list: List[float] = []
    terminated_list: List[bool] = []
    truncated_list: List[bool] = []
    completed_episodes: List[Dict[str, object]] = []

    for step_index in range(num_steps):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action_mask = env.action_mask()
        action_mask_tensor = torch.as_tensor(action_mask, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, raw_action, _, _, _ = model.sample_action(state_tensor, action_mask_tensor)
        next_state, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())

        state_list.append(np.asarray(state, dtype=np.float32).copy())
        next_state_list.append(np.asarray(next_state, dtype=np.float32).copy())
        raw_action_list.append(raw_action.squeeze(0).cpu().numpy())
        action_mask_list.append(action_mask)
        reward_list.append(float(reward))
        terminated_list.append(bool(terminated))
        truncated_list.append(bool(truncated))
        running_episode_return += float(reward)
        running_episode_length += 1

        if terminated or truncated:
            completed_episodes.append(
                {
                    "relative_steps": step_index + 1,
                    "episode_return": running_episode_return,
                    "episode_length": running_episode_length,
                    "info": info,
                }
            )
            state, _ = env.reset()
            running_episode_return = 0.0
            running_episode_length = 0
        else:
            state = next_state

    rollout = {
        "states": torch.as_tensor(np.asarray(state_list), dtype=torch.float32, device=device),
        "next_states": torch.as_tensor(np.asarray(next_state_list), dtype=torch.float32, device=device),
        "raw_actions": torch.as_tensor(np.asarray(raw_action_list), dtype=torch.float32, device=device),
        "action_masks": torch.as_tensor(np.asarray(action_mask_list), dtype=torch.float32, device=device),
        "rewards": torch.as_tensor(np.asarray(reward_list), dtype=torch.float32, device=device),
        "terminateds": torch.as_tensor(np.asarray(terminated_list), dtype=torch.float32, device=device),
        "truncateds": torch.as_tensor(np.asarray(truncated_list), dtype=torch.float32, device=device),
    }
    return rollout, state, running_episode_return, running_episode_length, completed_episodes


def compute_returns_and_advantages(
    model: ActorCritic,
    rollout: Dict[str, torch.Tensor],
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GAE with separate bootstrap and episode-continuation masks.

    ``bootstrap_mask = 1 - terminated`` means a timeout has a valid value
    bootstrap.  ``episode_continuation = 1 - terminated - truncated`` prevents
    advantages from leaking backward across the subsequent reset episode.
    """

    with torch.no_grad():
        _, _, values = model(rollout["states"])
        _, _, next_values = model(rollout["next_states"])

    terminateds = rollout["terminateds"]
    truncateds = rollout["truncateds"]
    bootstrap_mask = 1.0 - terminateds
    episode_continuation = 1.0 - torch.maximum(terminateds, truncateds)
    deltas = rollout["rewards"] + gamma * bootstrap_mask * next_values - values

    advantages = torch.zeros_like(deltas)
    gae = torch.zeros((), dtype=torch.float32, device=deltas.device)
    for step in reversed(range(deltas.shape[0])):
        gae = deltas[step] + gamma * gae_lambda * episode_continuation[step] * gae
        advantages[step] = gae
    returns = advantages + values
    normalized_advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    return returns, normalized_advantages


def compute_a2c_loss(
    model: ActorCritic,
    rollout: Dict[str, torch.Tensor],
    returns: torch.Tensor,
    advantages: torch.Tensor,
    hyperparameters: AgentHyperParameters,
):
    log_probs, entropy_estimates, values = model.evaluate_actions(
        rollout["states"], rollout["raw_actions"], rollout["action_masks"]
    )
    policy_loss = -(log_probs * advantages.detach()).mean()
    value_loss = (returns.detach() - values).pow(2).mean()
    entropy = entropy_estimates.mean()
    total_loss = policy_loss + hyperparameters.value_coef * value_loss - hyperparameters.entropy_coef * entropy
    return total_loss, policy_loss, value_loss, entropy


def evaluate(env, model: ActorCritic, config: ExperimentConfig, device: torch.device) -> Dict[str, float]:
    model.eval()

    def select_action(state: np.ndarray) -> np.ndarray:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = model.deterministic_action(state_tensor)
        return action.squeeze(0).cpu().numpy()

    try:
        return evaluate_episodes(env, select_action, config)
    finally:
        model.train()


def save_checkpoint(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    total_steps: int,
    best_success: float,
    best_min_net_distance: float,
    best_return: float,
    config: ExperimentConfig,
    checkpoint_path: Path,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "total_steps": total_steps,
            "best_success": best_success,
            "best_min_net_distance": best_min_net_distance,
            "best_return": best_return,
            "launch_curriculum_stage": config.launch_curriculum_stage,
            "curriculum_success_streak": config.curriculum_success_streak,
        },
        checkpoint_path,
    )


def load_checkpoint(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    checkpoint_path: Path,
    device: torch.device,
) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def prepare_run(config: ExperimentConfig, hyperparameters: AgentHyperParameters) -> Path:
    output_dir = config.agent_run_dir("a2c")
    prefix = "seed_{}".format(config.seed)
    core_names = (
        "{}_best.pt".format(prefix),
        "{}_stage0_best.pt".format(prefix),
        "{}_stage1_best.pt".format(prefix),
        "{}_last.pt".format(prefix),
        "{}_config.json".format(prefix),
        "{}_train_metrics.csv".format(prefix),
        "{}_eval_metrics.csv".format(prefix),
    )
    if config.resume:
        checkpoint_path = output_dir / "{}_last.pt".format(prefix)
        config_path = output_dir / "{}_config.json".format(prefix)
        if not checkpoint_path.exists() or not config_path.exists():
            raise FileNotFoundError("resume requires {} and {}".format(checkpoint_path, config_path))
        with config_path.open(encoding="utf-8") as file:
            saved = json.load(file)
        for name, value in saved["a2c"].items():
            if hasattr(hyperparameters, name):
                setattr(hyperparameters, name, value)
        return output_dir
    if any((output_dir / name).exists() for name in core_names) and not config.overwrite:
        raise FileExistsError("{} already has files for {}; choose a new seed or pass --overwrite".format(output_dir, prefix))
    if config.overwrite:
        for name in core_names:
            path = output_dir / name
            if path.exists():
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "{}_config.json".format(prefix)).open("w", encoding="utf-8") as file:
        json.dump({"environment": config.to_dict(), "a2c": hyperparameters.to_dict()}, file, indent=2)
    return output_dir


def train(config: ExperimentConfig):
    seed = config.seed
    set_seed(seed)
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else "cpu")
    if config.device != "auto":
        device = torch.device(config.device)
    hyperparameters = AgentHyperParameters()
    output_dir = prepare_run(config, hyperparameters)
    prefix = "seed_{}".format(seed)
    env = InterceptionEnv(config)
    eval_env = InterceptionEnv(config)
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = ActorCritic(
        observation_dim,
        action_dim,
        hyperparameters.hidden_dim,
        hyperparameters.min_log_std,
        hyperparameters.max_log_std,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters.learning_rate)

    train_log_path = output_dir / "{}_train_metrics.csv".format(prefix)
    eval_log_path = output_dir / "{}_eval_metrics.csv".format(prefix)
    total_steps = 0
    best_success = -float("inf")
    best_min_net_distance = float("inf")
    best_return = -float("inf")
    if config.resume:
        checkpoint = load_checkpoint(
            model,
            optimizer,
            output_dir / "{}_last.pt".format(prefix),
            device,
        )
        total_steps = int(checkpoint["total_steps"])
        best_success = float(checkpoint.get("best_success", -float("inf")))
        best_min_net_distance = float(checkpoint.get("best_min_net_distance", float("inf")))
        best_return = float(checkpoint.get("best_return", -float("inf")))
        config.launch_curriculum_stage = int(checkpoint.get("launch_curriculum_stage", 0))
        config.curriculum_success_streak = int(checkpoint.get("curriculum_success_streak", 0))
        training_end_step = total_steps + config.additional_train_steps if config.additional_train_steps > 0 else None
        destination = training_end_step if training_end_step is not None else "manual stop"
        print("resumed a2c {} at step {}; training to {}".format(config.mode, total_steps, destination))
    else:
        training_end_step = config.total_train_steps
    next_eval_step = (total_steps // config.eval_interval_steps + 1) * config.eval_interval_steps
    next_save_step = (total_steps // config.save_interval_steps + 1) * config.save_interval_steps
    state, _ = env.reset(seed=seed + total_steps)
    running_episode_return = 0.0
    running_episode_length = 0

    try:
        while training_end_step is None or total_steps < training_end_step:
            remaining_budget = hyperparameters.rollout_steps if training_end_step is None else training_end_step - total_steps
            until_eval = max(1, next_eval_step - total_steps)
            rollout_steps = min(hyperparameters.rollout_steps, remaining_budget, until_eval)
            rollout, state, running_episode_return, running_episode_length, completed_episodes = collect_rollout(
                env,
                model,
                state,
                rollout_steps,
                device,
                running_episode_return,
                running_episode_length,
            )
            returns, advantages = compute_returns_and_advantages(
                model,
                rollout,
                hyperparameters.gamma,
                hyperparameters.gae_lambda,
            )
            total_loss, policy_loss, value_loss, entropy = compute_a2c_loss(
                model,
                rollout,
                returns,
                advantages,
                hyperparameters,
            )
            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparameters.max_grad_norm))
            optimizer.step()
            model.clamp_log_std()

            previous_total_steps = total_steps
            total_steps += rollout_steps
            for episode in completed_episodes:
                append_csv_row(
                    train_log_path,
                    TRAIN_METRIC_FIELDS,
                    {
                        "total_steps": previous_total_steps + int(episode["relative_steps"]),
                        "episode_return": episode["episode_return"],
                        "episode_length": episode["episode_length"],
                        "actor_loss": float(policy_loss.item()),
                        "critic_loss": float(value_loss.item()),
                    },
                )

            if total_steps >= next_eval_step or (
                training_end_step is not None and total_steps == training_end_step
            ):
                evaluation = evaluate(eval_env, model, config, device)
                evaluation["total_steps"] = total_steps
                append_csv_row(eval_log_path, EVAL_METRIC_FIELDS, evaluation)
                print(
                    "steps: {} | stage: {} | success: {:.1%} | rule/RL aim: {:.1%}/{:.1%} | return: {:.2f} | "
                    "policy/value/entropy: {:.3f}/{:.3f}/{:.3f} | grad: {:.3g}".format(
                        total_steps,
                        config.launch_curriculum_name,
                        evaluation["success_rate"],
                        evaluation["rule_aim_rate"],
                        evaluation["rl_aim_rate"],
                        evaluation["mean_return"],
                        float(policy_loss.item()),
                        float(value_loss.item()),
                        float(entropy.item()),
                        grad_norm,
                    )
                )
                evaluation_min_net = evaluation["mean_min_net_distance"]
                finite_min_net = evaluation_min_net if np.isfinite(evaluation_min_net) else float("inf")
                better = (
                    evaluation["success_rate"] > best_success
                    or (evaluation["success_rate"] == best_success and finite_min_net < best_min_net_distance)
                    or (
                        evaluation["success_rate"] == best_success
                        and finite_min_net == best_min_net_distance
                        and evaluation["mean_return"] > best_return
                    )
                )
                if better:
                    best_success = evaluation["success_rate"]
                    best_min_net_distance = finite_min_net
                    best_return = evaluation["mean_return"]
                    save_checkpoint(
                        model, optimizer, total_steps, best_success, best_min_net_distance,
                        best_return, config, output_dir / "{}_best.pt".format(prefix),
                    )
                    save_checkpoint(
                        model, optimizer, total_steps, best_success, best_min_net_distance,
                        best_return, config,
                        output_dir / "{}_stage{}_best.pt".format(prefix, config.launch_curriculum_stage),
                    )
                if config.update_launch_curriculum(evaluation["success_rate"]):
                    print("aim curriculum advanced to {}".format(config.launch_curriculum_name))
                    best_success = -float("inf")
                    best_min_net_distance = float("inf")
                    best_return = -float("inf")
                    state, _ = env.reset()
                    running_episode_return, running_episode_length = 0.0, 0
                    save_checkpoint(
                        model, optimizer, total_steps, best_success, best_min_net_distance,
                        best_return, config, output_dir / "{}_last.pt".format(prefix),
                    )
                while next_eval_step <= total_steps:
                    next_eval_step += config.eval_interval_steps

            if total_steps >= next_save_step:
                save_checkpoint(
                    model, optimizer, total_steps, best_success, best_min_net_distance,
                    best_return, config, output_dir / "{}_last.pt".format(prefix),
                )
                while next_save_step <= total_steps:
                    next_save_step += config.save_interval_steps
    finally:
        save_checkpoint(
            model, optimizer, total_steps, best_success, best_min_net_distance,
            best_return, config, output_dir / "{}_last.pt".format(prefix),
        )
        env.close()
        eval_env.close()

    return model
