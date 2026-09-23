"""Small shared utilities; update rules stay in their own algorithm files.

The important convention in this module is that a Gymnasium transition keeps
``terminated`` and ``truncated`` as separate values.  A rollout stops on either
flag, but Bellman targets mask *only* ``terminated``.
"""

import csv
import random
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch

from config import ExperimentConfig


TRAIN_METRIC_FIELDS = [
    "total_steps",
    "episode_return",
    "episode_length",
    "actor_loss",
    "critic_loss",
]

EVAL_METRIC_FIELDS = [
    "total_steps",
    "curriculum_stage",
    "success_rate",
    "mean_return",
    "mean_capture_time",
    "mean_min_distance",
    "mean_min_net_distance",
    "mean_effort",
    "mean_launch_distance",
    "launch_rate",
    "rule_aim_rate",
    "rl_aim_rate",
    "mean_launch_time",
    "gate_open_rate",
    "mean_gate_open_steps",
    "mean_first_gate_time",
    "hit_rate",
    "miss_rate",
    "target_exit_rate",
    "interceptor_exit_rate",
]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch.  Environments receive their own seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_csv_row(path: Path, fieldnames: Sequence[str], row: Dict[str, object]) -> None:
    """Append one metric row and write a header only for a new log file."""

    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def evaluate_episodes(
    env,
    select_action: Callable[[np.ndarray], np.ndarray],
    config: ExperimentConfig,
) -> Dict[str, float]:
    """Evaluate deterministically on the common held-out seed bank.

    Evaluation transitions are intentionally not counted as training environment
    steps.  Every algorithm uses the same deterministic held-out episodes.
    """

    successes: List[float] = []
    returns: List[float] = []
    capture_times: List[float] = []
    min_distances: List[float] = []
    min_net_distances: List[float] = []
    efforts: List[float] = []
    launch_distances: List[float] = []
    launch_times: List[float] = []
    launch_used: List[float] = []
    rule_aims: List[float] = []
    rl_aims: List[float] = []
    gate_opened: List[float] = []
    gate_open_steps: List[float] = []
    first_gate_times: List[float] = []
    reasons: List[str] = []

    for scenario_seed in config.eval_seed_bank[: config.n_eval_episodes]:
        state, _ = env.reset(seed=int(scenario_seed))
        terminated = False
        truncated = False
        episode_return = 0.0
        info: Dict[str, object] = {}

        while not (terminated or truncated):
            action = select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)

        successes.append(float(info["success"]))
        returns.append(episode_return)
        capture_times.append(float(info["capture_time"]))
        min_distances.append(float(info["min_distance"]))
        min_net_distances.append(float(info["min_net_distance"]))
        efforts.append(float(info["control_effort"]))
        launch_distances.append(float(info["launch_distance"]))
        launch_times.append(float(info["launch_time"]))
        launch_used.append(float(info["launch_used"]))
        rule_aims.append(float(info["rule_aim"]))
        rl_aims.append(float(info["rl_aim"]))
        gate_opened.append(float(info["gate_ever_open"]))
        gate_open_steps.append(float(info["gate_open_steps"]))
        first_gate_times.append(float(info["first_gate_time"]))
        reasons.append(str(info["termination_reason"]))

    def finite_mean(values: List[float]) -> float:
        values_array = np.asarray(values, dtype=np.float64)
        finite_values = values_array[np.isfinite(values_array)]
        if finite_values.size:
            return float(np.mean(finite_values))
        return float("nan")

    return {
        "curriculum_stage": float(config.launch_curriculum_stage),
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "mean_capture_time": finite_mean(capture_times),
        "mean_min_distance": float(np.mean(min_distances)),
        "mean_min_net_distance": finite_mean(min_net_distances),
        "mean_effort": float(np.mean(efforts)),
        "mean_launch_distance": finite_mean(launch_distances),
        "launch_rate": float(np.mean(launch_used)),
        "rule_aim_rate": float(np.mean(rule_aims)),
        "rl_aim_rate": float(np.mean(rl_aims)),
        "mean_launch_time": finite_mean(launch_times),
        "gate_open_rate": float(np.mean(gate_opened)),
        "mean_gate_open_steps": float(np.mean(gate_open_steps)),
        "mean_first_gate_time": finite_mean(first_gate_times),
        "hit_rate": float(np.mean([reason == "hit" for reason in reasons])),
        "miss_rate": float(np.mean([reason == "miss" for reason in reasons])),
        "target_exit_rate": float(np.mean([reason == "target_exit" for reason in reasons])),
        "interceptor_exit_rate": float(np.mean([reason == "interceptor_exit" for reason in reasons])),
    }


class ReplayBuffer:
    """Replay memory that deliberately preserves both Gymnasium flags.

    ``next_states`` is always the state returned by ``env.step`` *before* a
    training loop calls ``reset``.  This makes timeout bootstrap correct:
    ``r + gamma * Q(next_state)`` for truncated transitions and no bootstrap
    only for true terminations.
    """

    def __init__(self, observation_dim: int, action_dim: int, capacity: int):
        self.states = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.terminateds = np.zeros(capacity, dtype=np.float32)
        self.truncateds = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.pointer = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_states[self.pointer] = next_state
        self.terminateds[self.pointer] = float(terminated)
        self.truncateds[self.pointer] = float(truncated)
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "states": torch.as_tensor(self.states[indices], dtype=torch.float32, device=device),
            "actions": torch.as_tensor(self.actions[indices], dtype=torch.float32, device=device),
            "rewards": torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device),
            "next_states": torch.as_tensor(self.next_states[indices], dtype=torch.float32, device=device),
            "terminateds": torch.as_tensor(self.terminateds[indices], dtype=torch.float32, device=device),
            "truncateds": torch.as_tensor(self.truncateds[indices], dtype=torch.float32, device=device),
        }

    def state_dict(self) -> Dict[str, object]:
        """Values required to continue an off-policy run from last.pt."""

        return {
            "states": torch.from_numpy(self.states[: self.size].copy()),
            "actions": torch.from_numpy(self.actions[: self.size].copy()),
            "rewards": torch.from_numpy(self.rewards[: self.size].copy()),
            "next_states": torch.from_numpy(self.next_states[: self.size].copy()),
            "terminateds": torch.from_numpy(self.terminateds[: self.size].copy()),
            "truncateds": torch.from_numpy(self.truncateds[: self.size].copy()),
            "pointer": self.pointer,
            "size": self.size,
        }

    def load_state_dict(self, values: Dict[str, object]) -> None:
        saved_size = int(values["size"])
        count = min(saved_size, self.capacity)
        self.states[:count] = values["states"][:count].cpu().numpy()
        self.actions[:count] = values["actions"][:count].cpu().numpy()
        self.rewards[:count] = values["rewards"][:count].cpu().numpy()
        self.next_states[:count] = values["next_states"][:count].cpu().numpy()
        self.terminateds[:count] = values["terminateds"][:count].cpu().numpy()
        self.truncateds[:count] = values["truncateds"][:count].cpu().numpy()
        self.size = count
        self.pointer = int(values["pointer"]) % self.capacity if count == self.capacity else count

    def __len__(self) -> int:
        return self.size
