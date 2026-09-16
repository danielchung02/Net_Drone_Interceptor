"""PPO for the RotorPy interception task.

This file intentionally contains the distribution, rollout, GAE, update,
evaluation, and checkpoint logic so PPO can be studied as one unit.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from agent.common import EVAL_METRIC_FIELDS, append_csv_row, evaluate_episodes, set_seed
from config import ExperimentConfig
from interception_env import InterceptionEnv


TRAIN_METRIC_FIELDS = [
    "total_steps",
    "episode_return",
    "episode_length",
    "policy_loss",
    "value_loss",
]


class PPOHyperParameters:
    """PPO-only values, deliberately written as explicit self assignments."""

    def __init__(self):
        self.hidden_dim = 128
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.rollout_steps = 1024
        self.epochs = 10
        self.minibatch_size = 128
        self.clip_coef = 0.20
        self.value_coef = 0.5
        self.entropy_coef = 1e-3
        self.max_grad_norm = 10.0

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.features(states)
        mean = self.actor_mean(features)
        return mean, self.log_std.expand_as(mean), self.critic(features).squeeze(-1)

    def sample_action(self, states: torch.Tensor):
        mean, log_std, values = self(states)
        distribution = Normal(mean, torch.exp(torch.clamp(log_std, -20.0, 2.0)))
        raw_actions = distribution.rsample()
        actions = torch.tanh(raw_actions)
        log_probs = (distribution.log_prob(raw_actions) - torch.log(1.0 - actions.pow(2) + 1e-6)).sum(-1)
        return actions, raw_actions, log_probs, -log_probs, values

    def evaluate_actions(self, states: torch.Tensor, raw_actions: torch.Tensor):
        mean, log_std, values = self(states)
        distribution = Normal(mean, torch.exp(torch.clamp(log_std, -20.0, 2.0)))
        actions = torch.tanh(raw_actions)
        log_probs = (distribution.log_prob(raw_actions) - torch.log(1.0 - actions.pow(2) + 1e-6)).sum(-1)
        return log_probs, -log_probs, values

    def deterministic_action(self, states: torch.Tensor) -> torch.Tensor:
        mean, _, _ = self(states)
        return torch.tanh(mean)


def _device(config: ExperimentConfig) -> torch.device:
    return torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else config.device if config.device != "auto" else "cpu")


def _prepare_run(config: ExperimentConfig, hyper: PPOHyperParameters) -> Path:
    run_dir = config.agent_run_dir("ppo")
    prefix = "seed_{}".format(config.seed)
    core_names = (
        "{}_best.pt".format(prefix),
        "{}_last.pt".format(prefix),
        "{}_config.json".format(prefix),
        "{}_train_metrics.csv".format(prefix),
        "{}_eval_metrics.csv".format(prefix),
    )
    if config.resume:
        checkpoint_path = run_dir / "{}_last.pt".format(prefix)
        config_path = run_dir / "{}_config.json".format(prefix)
        if not checkpoint_path.exists() or not config_path.exists():
            raise FileNotFoundError("resume requires {} and {}".format(checkpoint_path, config_path))
        with config_path.open(encoding="utf-8") as file:
            saved = json.load(file)
        for name, value in saved["ppo"].items():
            if hasattr(hyper, name):
                setattr(hyper, name, value)
        return run_dir
    if any((run_dir / name).exists() for name in core_names) and not config.overwrite:
        raise FileExistsError("{} already has files for {}; choose a new seed or pass --overwrite".format(run_dir, prefix))
    if config.overwrite:
        for name in core_names:
            path = run_dir / name
            if path.exists():
                path.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "{}_config.json".format(prefix)).open("w", encoding="utf-8") as file:
        json.dump({"environment": config.to_dict(), "ppo": hyper.to_dict()}, file, indent=2)
    return run_dir


def collect_rollout(env, model, state, count, device, episode_return, episode_length):
    """Store final next_state before reset; this preserves truncation bootstrap."""

    names = ("states", "next_states", "raw_actions", "old_log_probs", "rewards", "terminateds", "truncateds")
    data: Dict[str, List[object]] = {name: [] for name in names}
    completed: List[Dict[str, float]] = []
    for index in range(count):
        tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, raw_action, log_prob, _, _ = model.sample_action(tensor)
        next_state, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
        data["states"].append(np.asarray(state, dtype=np.float32))
        data["next_states"].append(np.asarray(next_state, dtype=np.float32))
        data["raw_actions"].append(raw_action.squeeze(0).cpu().numpy())
        data["old_log_probs"].append(float(log_prob.item()))
        data["rewards"].append(float(reward))
        data["terminateds"].append(float(terminated))
        data["truncateds"].append(float(truncated))
        episode_return += float(reward)
        episode_length += 1
        if terminated or truncated:
            completed.append({"relative_steps": index + 1, "episode_return": episode_return, "episode_length": episode_length})
            state, _ = env.reset()
            episode_return, episode_length = 0.0, 0
        else:
            state = next_state
    rollout = {name: torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device) for name, values in data.items()}
    return rollout, state, episode_return, episode_length, completed


def returns_and_advantages(model: ActorCritic, rollout: Dict[str, torch.Tensor], hyper: PPOHyperParameters):
    """GAE: terminal mask and rollout-continuation mask have different jobs."""

    with torch.no_grad():
        _, _, values = model(rollout["states"])
        _, _, next_values = model(rollout["next_states"])
    terminateds = rollout["terminateds"]
    truncateds = rollout["truncateds"]
    bootstrap_mask = 1.0 - terminateds
    continuation_mask = 1.0 - torch.maximum(terminateds, truncateds)
    deltas = rollout["rewards"] + hyper.gamma * bootstrap_mask * next_values - values
    advantages = torch.zeros_like(deltas)
    gae = torch.zeros((), device=deltas.device)
    for index in reversed(range(len(deltas))):
        gae = deltas[index] + hyper.gamma * hyper.gae_lambda * continuation_mask[index] * gae
        advantages[index] = gae
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    return returns, advantages


def update(model, optimizer, rollout, returns, advantages, hyper: PPOHyperParameters):
    policy_losses: List[float] = []
    value_losses: List[float] = []
    entropies: List[float] = []
    for _ in range(hyper.epochs):
        indices = torch.randperm(len(returns), device=returns.device)
        for start in range(0, len(returns), hyper.minibatch_size):
            batch = indices[start : start + hyper.minibatch_size]
            new_log_probs, entropy, values = model.evaluate_actions(rollout["states"][batch], rollout["raw_actions"][batch])
            ratio = torch.exp(new_log_probs - rollout["old_log_probs"][batch])
            surrogate = torch.minimum(
                ratio * advantages[batch],
                torch.clamp(ratio, 1.0 - hyper.clip_coef, 1.0 + hyper.clip_coef) * advantages[batch],
            )
            policy_loss = -surrogate.mean()
            value_loss = (returns[batch] - values).pow(2).mean()
            loss = policy_loss + hyper.value_coef * value_loss - hyper.entropy_coef * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyper.max_grad_norm)
            optimizer.step()
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.mean().item()))
    return {"policy_loss": float(np.mean(policy_losses)), "value_loss": float(np.mean(value_losses)), "entropy": float(np.mean(entropies))}


def evaluate(config: ExperimentConfig, model: ActorCritic, device: torch.device) -> Dict[str, float]:
    env = InterceptionEnv(config)
    model.eval()

    def select_action(state: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = model.deterministic_action(tensor)
        return action.squeeze(0).cpu().numpy()

    try:
        return evaluate_episodes(env, select_action, config)
    finally:
        env.close()
        model.train()


def _save(path: Path, model: ActorCritic, optimizer: optim.Optimizer, steps: int, best_success: float) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "steps": steps, "best_success": best_success}, path)


def load_last(path: Path, model: ActorCritic, optimizer: optim.Optimizer, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["steps"]), float(checkpoint.get("best_success", -float("inf")))


def train(config: ExperimentConfig) -> ActorCritic:
    hyper = PPOHyperParameters()
    set_seed(config.seed)
    device = _device(config)
    run_dir = _prepare_run(config, hyper)
    prefix = "seed_{}".format(config.seed)
    env = InterceptionEnv(config)
    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], hyper.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyper.learning_rate)
    total_steps, episode_return, episode_length = 0, 0.0, 0
    best_success = -float("inf")
    if config.resume:
        total_steps, best_success = load_last(
            run_dir / "{}_last.pt".format(prefix), model, optimizer, device
        )
        training_end_step = total_steps + config.additional_train_steps
        print("resumed ppo {} at step {}; training to {}".format(config.mode, total_steps, training_end_step))
    else:
        training_end_step = config.total_train_steps
    next_evaluation = (total_steps // config.eval_interval_steps + 1) * config.eval_interval_steps
    next_save = (total_steps // config.save_interval_steps + 1) * config.save_interval_steps
    state, _ = env.reset(seed=config.seed + total_steps)
    metrics = {"policy_loss": float("nan"), "value_loss": float("nan"), "entropy": float("nan")}
    try:
        while total_steps < training_end_step:
            count = min(hyper.rollout_steps, training_end_step - total_steps, max(1, next_evaluation - total_steps))
            rollout, state, episode_return, episode_length, completed = collect_rollout(env, model, state, count, device, episode_return, episode_length)
            returns, advantages = returns_and_advantages(model, rollout, hyper)
            metrics = update(model, optimizer, rollout, returns, advantages, hyper)
            previous_steps, total_steps = total_steps, total_steps + count
            for episode in completed:
                append_csv_row(run_dir / "{}_train_metrics.csv".format(prefix), TRAIN_METRIC_FIELDS, {
                    "total_steps": previous_steps + int(episode["relative_steps"]),
                    "episode_return": episode["episode_return"], "episode_length": episode["episode_length"],
                    "policy_loss": metrics["policy_loss"], "value_loss": metrics["value_loss"],
                })
            if total_steps >= next_evaluation or total_steps == training_end_step:
                result = evaluate(config, model, device)
                append_csv_row(
                    run_dir / "{}_eval_metrics.csv".format(prefix),
                    EVAL_METRIC_FIELDS,
                    {"total_steps": total_steps, **result},
                )
                print("steps={} success={:.1%} return={:.2f} ppo={:.3f}/{:.3f}".format(
                    total_steps, result["success_rate"], result["mean_return"], metrics["policy_loss"], metrics["value_loss"]
                ))
                if result["success_rate"] > best_success:
                    best_success = result["success_rate"]
                    _save(run_dir / "{}_best.pt".format(prefix), model, optimizer, total_steps, best_success)
                while next_evaluation <= total_steps:
                    next_evaluation += config.eval_interval_steps
            if total_steps >= next_save:
                _save(run_dir / "{}_last.pt".format(prefix), model, optimizer, total_steps, best_success)
                while next_save <= total_steps:
                    next_save += config.save_interval_steps
    finally:
        _save(run_dir / "{}_last.pt".format(prefix), model, optimizer, total_steps, best_success)
        env.close()
    return model
