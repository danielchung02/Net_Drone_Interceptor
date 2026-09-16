"""Deep Deterministic Policy Gradient for normalized continuous actions."""

import copy
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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


class AgentHyperParameters:
    """DDPG choices, separate from the shared physical experiment settings."""

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
        self.exploration_noise_std = 0.10
        self.max_grad_norm = 10.0

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


class Actor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # Environment action space is standardized to [-1, 1] in every
        # physical dimension, so tanh is the correct affine action mapping.
        return torch.tanh(self.net(states))


class Critic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        state_actions = torch.cat([states, actions], dim=-1)
        return self.net(state_actions).squeeze(dim=-1)


class DDPGAgent:
    def __init__(self, observation_dim: int, action_dim: int, hyperparameters: AgentHyperParameters, device: torch.device):
        self.device = device
        self.hyperparameters = hyperparameters
        self.actor = Actor(observation_dim, action_dim, hyperparameters.hidden_dim).to(device)
        self.critic = Critic(observation_dim, action_dim, hyperparameters.hidden_dim).to(device)
        self.target_actor = copy.deepcopy(self.actor).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=hyperparameters.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=hyperparameters.critic_learning_rate)
        for parameter in self.target_actor.parameters():
            parameter.requires_grad_(False)
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)

    def select_action(self, state: np.ndarray, add_noise: bool) -> np.ndarray:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor).squeeze(0).cpu().numpy()
        if add_noise:
            action = action + np.random.normal(
                0.0,
                self.hyperparameters.exploration_noise_std,
                size=action.shape,
            ).astype(np.float32)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    @torch.no_grad()
    def soft_update(self, online_network: nn.Module, target_network: nn.Module) -> None:
        for online_parameter, target_parameter in zip(online_network.parameters(), target_network.parameters()):
            target_parameter.mul_(1.0 - self.hyperparameters.tau).add_(self.hyperparameters.tau * online_parameter)

    def update_ddpg(self, replay_buffer: ReplayBuffer) -> Dict[str, float]:
        batch = replay_buffer.sample(self.hyperparameters.batch_size, self.device)
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        terminateds = batch["terminateds"]

        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            target_qvalues = self.target_critic(next_states, target_actions)
            # Do not use ``terminated or truncated`` here.  A timeout is a
            # time-limit cut and must bootstrap from replay's stored next_state.
            targets = rewards + self.hyperparameters.gamma * (1.0 - terminateds) * target_qvalues

        qvalues = self.critic(states, actions)
        critic_loss = (targets - qvalues).pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = float(torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.hyperparameters.max_grad_norm))
        self.critic_optimizer.step()

        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.hyperparameters.max_grad_norm))
        self.actor_optimizer.step()
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)
        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "q_mean": float(qvalues.mean().item()),
            "target_q_mean": float(targets.mean().item()),
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
        }


def evaluate(env, agent: DDPGAgent, config: ExperimentConfig) -> Dict[str, float]:
    agent.actor.eval()
    try:
        return evaluate_episodes(env, lambda state: agent.select_action(state, add_noise=False), config)
    finally:
        agent.actor.train()


def save_checkpoint(
    agent: DDPGAgent,
    total_steps: int,
    best_success: float,
    checkpoint_path: Path,
    replay_buffer=None,
) -> None:
    checkpoint = {
        "actor_state_dict": agent.actor.state_dict(),
        "critic_state_dict": agent.critic.state_dict(),
        "target_actor_state_dict": agent.target_actor.state_dict(),
        "target_critic_state_dict": agent.target_critic.state_dict(),
        "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": agent.critic_optimizer.state_dict(),
        "total_steps": total_steps,
        "best_success": best_success,
    }
    if replay_buffer is not None:
        checkpoint["replay_buffer"] = replay_buffer.state_dict()
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(agent: DDPGAgent, checkpoint_path: Path, device: torch.device, replay_buffer=None) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.critic.load_state_dict(checkpoint["critic_state_dict"])
    agent.target_actor.load_state_dict(checkpoint["target_actor_state_dict"])
    agent.target_critic.load_state_dict(checkpoint["target_critic_state_dict"])
    agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
    if replay_buffer is not None and "replay_buffer" in checkpoint:
        replay_buffer.load_state_dict(checkpoint["replay_buffer"])
    return checkpoint


def prepare_run(config: ExperimentConfig, hyperparameters: AgentHyperParameters) -> Path:
    output_dir = config.agent_run_dir("ddpg")
    prefix = "seed_{}".format(config.seed)
    core_names = (
        "{}_best.pt".format(prefix),
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
        for name, value in saved["ddpg"].items():
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
        json.dump({"environment": config.to_dict(), "ddpg": hyperparameters.to_dict()}, file, indent=2)
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
    agent = DDPGAgent(observation_dim, action_dim, hyperparameters, device)
    replay_buffer = ReplayBuffer(observation_dim, action_dim, hyperparameters.replay_capacity)

    train_log_path = output_dir / "{}_train_metrics.csv".format(prefix)
    eval_log_path = output_dir / "{}_eval_metrics.csv".format(prefix)
    starting_steps = 0
    best_success = -float("inf")
    if config.resume:
        checkpoint = load_checkpoint(
            agent,
            output_dir / "{}_last.pt".format(prefix),
            device,
            replay_buffer,
        )
        starting_steps = int(checkpoint["total_steps"])
        best_success = float(checkpoint.get("best_success", -float("inf")))
        training_end_step = starting_steps + config.additional_train_steps
        print("resumed ddpg {} at step {}; training to {}".format(config.mode, starting_steps, training_end_step))
    else:
        training_end_step = config.total_train_steps
    metrics = {
        "actor_loss": float("nan"),
        "critic_loss": float("nan"),
        "q_mean": float("nan"),
        "target_q_mean": float("nan"),
        "actor_grad_norm": float("nan"),
        "critic_grad_norm": float("nan"),
    }
    state, _ = env.reset(seed=seed + starting_steps)
    episode_return = 0.0
    episode_length = 0

    try:
        total_steps = starting_steps
        for total_steps in range(starting_steps + 1, training_end_step + 1):
            if total_steps <= hyperparameters.start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, add_noise=True)

            # Store this next_state first.  If the transition is truncated,
            # replay must retain it for Bellman bootstrap before calling reset.
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
                    metrics = agent.update_ddpg(replay_buffer)

            if total_steps % config.eval_interval_steps == 0 or total_steps == training_end_step:
                evaluation = evaluate(eval_env, agent, config)
                evaluation["total_steps"] = total_steps
                append_csv_row(eval_log_path, EVAL_METRIC_FIELDS, evaluation)
                print(
                    "steps: {} | success: {:.1%} | return: {:.2f} | "
                    "actor/critic: {:.3f}/{:.3f} | Q/target: {:.2f}/{:.2f}".format(
                        total_steps,
                        evaluation["success_rate"],
                        evaluation["mean_return"],
                        metrics["actor_loss"],
                        metrics["critic_loss"],
                        metrics["q_mean"],
                        metrics["target_q_mean"],
                    )
                )
                if evaluation["success_rate"] > best_success:
                    best_success = evaluation["success_rate"]
                    save_checkpoint(agent, total_steps, best_success, output_dir / "{}_best.pt".format(prefix))

            if total_steps % config.save_interval_steps == 0:
                save_checkpoint(agent, total_steps, best_success, output_dir / "{}_last.pt".format(prefix), replay_buffer)
    finally:
        save_checkpoint(agent, total_steps, best_success, output_dir / "{}_last.pt".format(prefix), replay_buffer)
        env.close()
        eval_env.close()

    return agent
