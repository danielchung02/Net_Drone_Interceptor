"""Soft Actor-Critic with twin critics and optional automatic temperature."""

import copy
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from agent.common import (
    EVAL_METRIC_FIELDS,
    TRAIN_METRIC_FIELDS,
    ReplayBuffer,
    append_csv_row,
    evaluate_episodes,
    set_seed,
)
from config import ExperimentConfig
from interception_env import InterceptionEnv


def active_action_masks(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    masks = torch.ones_like(actions)
    gate_open = states[..., -2] > 0.5
    if actions.shape[-1] == 2:
        masks[...] = gate_open.unsqueeze(-1).to(actions.dtype)
    else:
        rl_aim_stage = states[..., -1] > 0.5
        aim_active = (gate_open & rl_aim_stage).unsqueeze(-1).to(actions.dtype)
        masks[..., 3:5] = aim_active
    return masks


def mask_inactive_aim_numpy(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).copy()
    if action.shape[-1] == 2:
        if state[-2] <= 0.5:
            action[:] = 0.0
    elif state[-1] <= 0.5 or state[-2] <= 0.5:
        action[3:5] = 0.0
    return action


class AgentHyperParameters:
    """SAC choices, including the entropy-temperature mechanism."""

    def __init__(self):
        self.hidden_dim = 128
        self.gamma = 0.99
        self.replay_capacity = 200_000
        self.batch_size = 256
        self.start_steps = 10_000
        self.update_after = 1_000
        self.updates_per_step = 1
        self.actor_learning_rate = 3e-4
        self.critic_learning_rate = 3e-4
        self.tau = 0.005
        self.alpha = 0.20
        self.auto_entropy_tuning = True
        self.alpha_learning_rate = 3e-4
        self.max_grad_norm = 10.0

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


class GaussianActor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.net(states)
        mean = self.mean_layer(features)
        log_std = torch.clamp(self.log_std_layer(features), min=-20.0, max=2.0)
        return mean, log_std

    def sample_action(self, states: torch.Tensor):
        mean, log_std = self.forward(states)
        distribution = Normal(mean, torch.exp(log_std))
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)
        log_probability = distribution.log_prob(raw_action) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_probability

    def deterministic_action(self, states: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(states)
        return torch.tanh(mean)


class TwinCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        input_dim = observation_dim + action_dim
        self.q1_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor):
        state_actions = torch.cat([states, actions], dim=-1)
        return (
            self.q1_network(state_actions).squeeze(dim=-1),
            self.q2_network(state_actions).squeeze(dim=-1),
        )


class SACAgent:
    def __init__(self, observation_dim: int, action_dim: int, hyperparameters: AgentHyperParameters, device: torch.device):
        self.device = device
        self.hyperparameters = hyperparameters
        self.action_dim = action_dim
        self.actor = GaussianActor(observation_dim, action_dim, hyperparameters.hidden_dim).to(device)
        self.critic = TwinCritic(observation_dim, action_dim, hyperparameters.hidden_dim).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=hyperparameters.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=hyperparameters.critic_learning_rate)
        if hyperparameters.auto_entropy_tuning:
            self.log_alpha = torch.tensor(
                np.log(hyperparameters.alpha),
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=hyperparameters.alpha_learning_rate)
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)

    @property
    def alpha(self) -> torch.Tensor:
        if self.log_alpha is None:
            return torch.tensor(self.hyperparameters.alpha, dtype=torch.float32, device=self.device)
        return self.log_alpha.exp()

    def select_action(self, state: np.ndarray, deterministic: bool) -> np.ndarray:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic_action(state_tensor)
            else:
                action, _ = self.actor.sample_action(state_tensor)
        action = action.squeeze(0).cpu().numpy().astype(np.float32)
        return mask_inactive_aim_numpy(state, action)

    @torch.no_grad()
    def soft_update(self) -> None:
        for online_parameter, target_parameter in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_parameter.mul_(1.0 - self.hyperparameters.tau).add_(self.hyperparameters.tau * online_parameter)

    def update_sac(self, replay_buffer: ReplayBuffer) -> Dict[str, float]:
        batch = replay_buffer.sample(self.hyperparameters.batch_size, self.device)
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        terminateds = batch["terminateds"]

        with torch.no_grad():
            next_actions, next_log_probability_per_dimension = self.actor.sample_action(next_states)
            next_action_masks = active_action_masks(next_states, next_actions)
            next_actions = next_actions * next_action_masks
            next_log_probabilities = (next_log_probability_per_dimension * next_action_masks).sum(dim=-1)
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            target_qvalues = torch.minimum(target_q1, target_q2) - self.alpha.detach() * next_log_probabilities
            # Only physical/MDP terminal events disable the bootstrap.  A
            # timeout keeps the real final next state saved in replay.
            targets = rewards + self.hyperparameters.gamma * (1.0 - terminateds) * target_qvalues

        q1, q2 = self.critic(states, actions)
        critic_loss = (targets - q1).pow(2).mean() + (targets - q2).pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = float(torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.hyperparameters.max_grad_norm))
        self.critic_optimizer.step()

        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        sampled_actions, log_probability_per_dimension = self.actor.sample_action(states)
        action_masks = active_action_masks(states, sampled_actions)
        sampled_actions = sampled_actions * action_masks
        log_probabilities = (log_probability_per_dimension * action_masks).sum(dim=-1)
        q1_pi, q2_pi = self.critic(states, sampled_actions)
        actor_loss = (self.alpha.detach() * log_probabilities - torch.minimum(q1_pi, q2_pi)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.hyperparameters.max_grad_norm))
        self.actor_optimizer.step()
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)

        alpha_loss = float("nan")
        if self.log_alpha is not None and self.alpha_optimizer is not None:
            target_entropy = -action_masks.sum(dim=-1)
            alpha_loss_tensor = -(
                self.log_alpha * (log_probabilities + target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss_tensor.backward()
            self.alpha_optimizer.step()
            alpha_loss = float(alpha_loss_tensor.item())

        self.soft_update()
        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "alpha_loss": alpha_loss,
            "alpha": float(self.alpha.detach().item()),
            "q_mean": float(torch.minimum(q1, q2).mean().item()),
            "target_q_mean": float(targets.mean().item()),
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
        }


def evaluate(env, agent: SACAgent, config: ExperimentConfig) -> Dict[str, float]:
    agent.actor.eval()
    try:
        return evaluate_episodes(env, lambda state: agent.select_action(state, deterministic=True), config)
    finally:
        agent.actor.train()


def save_checkpoint(
    agent: SACAgent,
    total_steps: int,
    best_success: float,
    best_min_net_distance: float,
    best_return: float,
    config: ExperimentConfig,
    checkpoint_path: Path,
    replay_buffer=None,
) -> None:
    checkpoint = {
        "actor_state_dict": agent.actor.state_dict(),
        "critic_state_dict": agent.critic.state_dict(),
        "target_critic_state_dict": agent.target_critic.state_dict(),
        "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": agent.critic_optimizer.state_dict(),
        "total_steps": total_steps,
        "best_success": best_success,
        "best_min_net_distance": best_min_net_distance,
        "best_return": best_return,
        "launch_curriculum_stage": config.launch_curriculum_stage,
        "curriculum_success_streak": config.curriculum_success_streak,
    }
    if agent.log_alpha is not None and agent.alpha_optimizer is not None:
        checkpoint["log_alpha"] = agent.log_alpha.detach().cpu()
        checkpoint["alpha_optimizer_state_dict"] = agent.alpha_optimizer.state_dict()
    if replay_buffer is not None:
        checkpoint["replay_buffer"] = replay_buffer.state_dict()
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(agent: SACAgent, checkpoint_path: Path, device: torch.device, replay_buffer=None) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.critic.load_state_dict(checkpoint["critic_state_dict"])
    agent.target_critic.load_state_dict(checkpoint["target_critic_state_dict"])
    agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
    if agent.log_alpha is not None and "log_alpha" in checkpoint:
        with torch.no_grad():
            agent.log_alpha.copy_(checkpoint["log_alpha"].to(device))
        agent.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
    if replay_buffer is not None and "replay_buffer" in checkpoint:
        replay_buffer.load_state_dict(checkpoint["replay_buffer"])
    return checkpoint


def prepare_run(config: ExperimentConfig, hyperparameters: AgentHyperParameters) -> Path:
    output_dir = config.agent_run_dir("sac")
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
        for name, value in saved["sac"].items():
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
        json.dump({"environment": config.to_dict(), "sac": hyperparameters.to_dict()}, file, indent=2)
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
    agent = SACAgent(observation_dim, action_dim, hyperparameters, device)
    replay_buffer = ReplayBuffer(observation_dim, action_dim, hyperparameters.replay_capacity)

    train_log_path = output_dir / "{}_train_metrics.csv".format(prefix)
    eval_log_path = output_dir / "{}_eval_metrics.csv".format(prefix)
    starting_steps = 0
    best_success = -float("inf")
    best_min_net_distance = float("inf")
    best_return = -float("inf")
    if config.resume:
        checkpoint = load_checkpoint(
            agent,
            output_dir / "{}_last.pt".format(prefix),
            device,
            replay_buffer,
        )
        starting_steps = int(checkpoint["total_steps"])
        best_success = float(checkpoint.get("best_success", -float("inf")))
        best_min_net_distance = float(checkpoint.get("best_min_net_distance", float("inf")))
        best_return = float(checkpoint.get("best_return", -float("inf")))
        config.launch_curriculum_stage = int(checkpoint.get("launch_curriculum_stage", 0))
        config.curriculum_success_streak = int(checkpoint.get("curriculum_success_streak", 0))
        training_end_step = starting_steps + config.additional_train_steps if config.additional_train_steps > 0 else None
        destination = training_end_step if training_end_step is not None else "manual stop"
        print("resumed sac {} at step {}; training to {}".format(config.mode, starting_steps, destination))
    else:
        training_end_step = config.total_train_steps
    metrics = {
        "actor_loss": float("nan"),
        "critic_loss": float("nan"),
        "alpha_loss": float("nan"),
        "alpha": hyperparameters.alpha,
        "q_mean": float("nan"),
        "target_q_mean": float("nan"),
    }
    state, _ = env.reset(seed=seed + starting_steps)
    episode_return = 0.0
    episode_length = 0

    try:
        total_steps = starting_steps
        while training_end_step is None or total_steps < training_end_step:
            total_steps += 1
            if total_steps <= hyperparameters.start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, deterministic=False)
            action = mask_inactive_aim_numpy(state, action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, terminated, truncated)
            episode_return += float(reward)
            episode_length += 1

            if terminated or truncated:
                append_csv_row(
                    train_log_path,
                    TRAIN_METRIC_FIELDS,
                    {
                        "total_steps": total_steps,
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                        "actor_loss": metrics["actor_loss"],
                        "critic_loss": metrics["critic_loss"],
                    },
                )
                state, _ = env.reset()
                episode_return = 0.0
                episode_length = 0
            else:
                state = next_state

            if total_steps >= hyperparameters.update_after and len(replay_buffer) >= hyperparameters.batch_size:
                for _ in range(hyperparameters.updates_per_step):
                    metrics = agent.update_sac(replay_buffer)

            if total_steps % config.eval_interval_steps == 0 or (
                training_end_step is not None and total_steps == training_end_step
            ):
                evaluation = evaluate(eval_env, agent, config)
                evaluation["total_steps"] = total_steps
                append_csv_row(eval_log_path, EVAL_METRIC_FIELDS, evaluation)
                print(
                    "steps: {} | stage: {} | success: {:.1%} | rule/RL aim: {:.1%}/{:.1%} | return: {:.2f} | "
                    "actor/critic: {:.3f}/{:.3f} | alpha: {:.3f}".format(
                        total_steps,
                        config.launch_curriculum_name,
                        evaluation["success_rate"],
                        evaluation["rule_aim_rate"],
                        evaluation["rl_aim_rate"],
                        evaluation["mean_return"],
                        metrics["actor_loss"],
                        metrics["critic_loss"],
                        metrics["alpha"],
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
                        agent, total_steps, best_success, best_min_net_distance,
                        best_return, config, output_dir / "{}_best.pt".format(prefix),
                    )
                    save_checkpoint(
                        agent, total_steps, best_success, best_min_net_distance,
                        best_return, config,
                        output_dir / "{}_stage{}_best.pt".format(prefix, config.launch_curriculum_stage),
                    )
                if config.update_launch_curriculum(evaluation["success_rate"]):
                    print("aim curriculum advanced to {}".format(config.launch_curriculum_name))
                    best_success = -float("inf")
                    best_min_net_distance = float("inf")
                    best_return = -float("inf")
                    replay_buffer = ReplayBuffer(observation_dim, action_dim, hyperparameters.replay_capacity)
                    state, _ = env.reset()
                    episode_return, episode_length = 0.0, 0
                    save_checkpoint(
                        agent, total_steps, best_success, best_min_net_distance,
                        best_return, config, output_dir / "{}_last.pt".format(prefix), replay_buffer,
                    )

            if total_steps % config.save_interval_steps == 0:
                save_checkpoint(
                    agent, total_steps, best_success, best_min_net_distance,
                    best_return, config, output_dir / "{}_last.pt".format(prefix), replay_buffer,
                )
    finally:
        save_checkpoint(
            agent, total_steps, best_success, best_min_net_distance,
            best_return, config, output_dir / "{}_last.pt".format(prefix), replay_buffer,
        )
        env.close()
        eval_env.close()

    return agent
