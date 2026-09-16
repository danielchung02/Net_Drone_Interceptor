"""REINFORCE with a learned state-value baseline for continuous actions."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
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


class AgentHyperParameters:
    """REINFORCE-specific settings; keep them next to its update equation."""

    def __init__(self):
        self.hidden_dim = 128
        self.gamma = 0.99
        self.policy_learning_rate = 3e-4
        self.baseline_learning_rate = 1e-3
        self.entropy_coef = 1e-3
        self.max_grad_norm = 10.0


def sample_squashed_normal(mean: torch.Tensor, log_std: torch.Tensor):
    """Reparameterized tanh-Gaussian action and its corrected log-probability."""

    distribution = Normal(mean, torch.exp(torch.clamp(log_std, -20.0, 2.0)))
    raw_action = distribution.rsample()
    action = torch.tanh(raw_action)
    log_prob = (distribution.log_prob(raw_action) - torch.log(1.0 - action.pow(2) + 1e-6)).sum(dim=-1)
    return action, raw_action, log_prob, -log_prob


def evaluate_squashed_normal(mean: torch.Tensor, log_std: torch.Tensor, raw_action: torch.Tensor):
    distribution = Normal(mean, torch.exp(torch.clamp(log_std, -20.0, 2.0)))
    action = torch.tanh(raw_action)
    log_prob = (distribution.log_prob(raw_action) - torch.log(1.0 - action.pow(2) + 1e-6)).sum(dim=-1)
    return action, log_prob, -log_prob


class PolicyNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.network(states)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std

    def sample_action(self, states: torch.Tensor):
        mean, log_std = self.forward(states)
        return sample_squashed_normal(mean, log_std)

    def evaluate_actions(self, states: torch.Tensor, raw_actions: torch.Tensor):
        mean, log_std = self.forward(states)
        return evaluate_squashed_normal(mean, log_std, raw_actions)

    def deterministic_action(self, states: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(states)
        return torch.tanh(mean)


class BaselineNetwork(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states).squeeze(dim=-1)


def collect_episode_fragment(
    env,
    policy: PolicyNetwork,
    baseline: BaselineNetwork,
    state: np.ndarray,
    max_steps: int,
    device: torch.device,
):
    """Collect at most ``max_steps`` without ever bootstrapping from reset()."""

    states: List[np.ndarray] = []
    raw_actions: List[np.ndarray] = []
    rewards: List[float] = []
    terminated = False
    truncated = False
    info: Dict[str, object] = {}
    next_state = state

    for _ in range(max_steps):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, raw_action, _, _ = policy.sample_action(state_tensor)
        action_for_env = action.squeeze(0).cpu().numpy()
        next_state, reward, terminated, truncated, info = env.step(action_for_env)
        states.append(np.asarray(state, dtype=np.float32).copy())
        raw_actions.append(raw_action.squeeze(0).cpu().numpy())
        rewards.append(float(reward))
        state = next_state
        if terminated or truncated:
            break

    # ``terminated`` is an MDP end.  Both Gymnasium truncation and an ordinary
    # rollout fragment have a real final next state whose value is valid.
    with torch.no_grad():
        if terminated:
            bootstrap_value = 0.0
        else:
            next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            bootstrap_value = float(baseline(next_state_tensor).item())

    return {
        "states": states,
        "raw_actions": raw_actions,
        "rewards": rewards,
        "next_state": next_state,
        "terminated": terminated,
        "truncated": truncated,
        "bootstrap_value": bootstrap_value,
        "info": info,
    }


def compute_returns(rewards: List[float], gamma: float, bootstrap_value: float) -> torch.Tensor:
    """Monte-Carlo returns with a critic bootstrap at a nonterminal cut."""

    returns: List[float] = []
    running_return = float(bootstrap_value)
    for reward in reversed(rewards):
        running_return = float(reward) + gamma * running_return
        returns.append(running_return)
    returns.reverse()
    return torch.as_tensor(returns, dtype=torch.float32)


def reinforce_update(
    policy: PolicyNetwork,
    baseline: BaselineNetwork,
    policy_optimizer: optim.Optimizer,
    baseline_optimizer: optim.Optimizer,
    fragment: Dict[str, object],
    hyperparameters: AgentHyperParameters,
    device: torch.device,
) -> Dict[str, float]:
    states_tensor = torch.as_tensor(np.asarray(fragment["states"]), dtype=torch.float32, device=device)
    raw_actions_tensor = torch.as_tensor(np.asarray(fragment["raw_actions"]), dtype=torch.float32, device=device)
    returns_tensor = compute_returns(
        fragment["rewards"],
        hyperparameters.gamma,
        float(fragment["bootstrap_value"]),
    ).to(device)

    _, log_probs, entropy_estimates = policy.evaluate_actions(states_tensor, raw_actions_tensor)
    baseline_values = baseline(states_tensor)
    advantages = returns_tensor - baseline_values.detach()
    # This keeps the user's original gamma^t REINFORCE weighting while a
    # learned V(s) reduces variance.
    time_steps = torch.arange(len(fragment["rewards"]), dtype=torch.float32, device=device)
    discount_weights = hyperparameters.gamma**time_steps
    policy_loss = -(discount_weights * advantages * log_probs).mean()
    policy_loss = policy_loss - hyperparameters.entropy_coef * entropy_estimates.mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), hyperparameters.max_grad_norm))
    policy_optimizer.step()

    baseline_loss = (returns_tensor - baseline_values).pow(2).mean()
    baseline_optimizer.zero_grad()
    baseline_loss.backward()
    baseline_grad_norm = float(torch.nn.utils.clip_grad_norm_(baseline.parameters(), hyperparameters.max_grad_norm))
    baseline_optimizer.step()

    return {
        "actor_loss": float(policy_loss.item()),
        "critic_loss": float(baseline_loss.item()),
        "policy_grad_norm": policy_grad_norm,
        "baseline_grad_norm": baseline_grad_norm,
    }


def evaluate(env, policy: PolicyNetwork, config: ExperimentConfig, device: torch.device) -> Dict[str, float]:
    policy.eval()

    def select_action(state: np.ndarray) -> np.ndarray:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = policy.deterministic_action(state_tensor)
        return action.squeeze(0).cpu().numpy()

    try:
        return evaluate_episodes(env, select_action, config)
    finally:
        policy.train()


def save_checkpoint(
    policy: PolicyNetwork,
    baseline: BaselineNetwork,
    policy_optimizer: optim.Optimizer,
    baseline_optimizer: optim.Optimizer,
    total_steps: int,
    best_mission_cost: float,
    checkpoint_path: Path,
) -> None:
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "baseline_state_dict": baseline.state_dict(),
            "policy_optimizer_state_dict": policy_optimizer.state_dict(),
            "baseline_optimizer_state_dict": baseline_optimizer.state_dict(),
            "total_steps": total_steps,
            "best_mission_cost": best_mission_cost,
        },
        checkpoint_path,
    )


def load_checkpoint(
    policy: PolicyNetwork,
    baseline: BaselineNetwork,
    policy_optimizer: optim.Optimizer,
    baseline_optimizer: optim.Optimizer,
    checkpoint_path: Path,
    device: torch.device,
) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    baseline.load_state_dict(checkpoint["baseline_state_dict"])
    policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
    baseline_optimizer.load_state_dict(checkpoint["baseline_optimizer_state_dict"])
    return checkpoint


def train(env, config: ExperimentConfig, seed: int):
    """Train one PN or E2E model; mode only changes environment action size."""

    set_seed(seed)
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else "cpu")
    if config.device != "auto":
        device = torch.device(config.device)
    hyperparameters = AgentHyperParameters()
    output_dir = config.agent_run_dir("reinforce")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "seed_{}".format(seed)
    from env.interception_env import InterceptionEnv
    eval_env = InterceptionEnv(config=config, render_mode="none")
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = PolicyNetwork(observation_dim, action_dim, hyperparameters.hidden_dim).to(device)
    baseline = BaselineNetwork(observation_dim, hyperparameters.hidden_dim).to(device)
    policy_optimizer = optim.Adam(policy.parameters(), lr=hyperparameters.policy_learning_rate)
    baseline_optimizer = optim.Adam(baseline.parameters(), lr=hyperparameters.baseline_learning_rate)

    train_log_path = output_dir / "{}_train_metrics.csv".format(prefix)
    eval_log_path = output_dir / "{}_eval_metrics.csv".format(prefix)
    total_steps = 0
    next_eval_step = config.eval_interval_steps
    next_save_step = config.save_interval_steps
    best_mission_cost = float("inf")
    state, _ = env.reset(seed=seed)
    episode_return = 0.0
    episode_length = 0

    try:
        while total_steps < config.total_train_steps:
            steps_to_eval = max(1, next_eval_step - total_steps)
            steps_to_budget = config.total_train_steps - total_steps
            fragment = collect_episode_fragment(
                env,
                policy,
                baseline,
                state,
                min(steps_to_eval, steps_to_budget),
                device,
            )
            metrics = reinforce_update(
                policy,
                baseline,
                policy_optimizer,
                baseline_optimizer,
                fragment,
                hyperparameters,
                device,
            )
            fragment_steps = len(fragment["rewards"])
            total_steps += fragment_steps
            episode_return += float(np.sum(fragment["rewards"]))
            episode_length += fragment_steps
            state = fragment["next_state"]

            if fragment["terminated"] or fragment["truncated"]:
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

            if total_steps >= next_eval_step or total_steps == config.total_train_steps:
                evaluation = evaluate(eval_env, policy, config, device)
                evaluation["total_steps"] = total_steps
                append_csv_row(eval_log_path, EVAL_METRIC_FIELDS, evaluation)
                print(
                    "steps: {} | mission cost: {:.3f} | capture: {:.1%} | "
                    "policy/baseline loss: {:.3f}/{:.3f}".format(
                        total_steps,
                        evaluation["mean_mission_cost"],
                        evaluation["capture_rate"],
                        metrics["actor_loss"],
                        metrics["critic_loss"],
                    )
                )
                if evaluation["mean_mission_cost"] < best_mission_cost:
                    best_mission_cost = evaluation["mean_mission_cost"]
                    save_checkpoint(
                        policy,
                        baseline,
                        policy_optimizer,
                        baseline_optimizer,
                        total_steps,
                        best_mission_cost,
                        output_dir / "{}_best.pt".format(prefix),
                    )
                while next_eval_step <= total_steps:
                    next_eval_step += config.eval_interval_steps

            if total_steps >= next_save_step:
                save_checkpoint(
                    policy,
                    baseline,
                    policy_optimizer,
                    baseline_optimizer,
                    total_steps,
                    best_mission_cost,
                    output_dir / "{}_last.pt".format(prefix),
                )
                while next_save_step <= total_steps:
                    next_save_step += config.save_interval_steps
    finally:
        save_checkpoint(
            policy,
            baseline,
            policy_optimizer,
            baseline_optimizer,
            total_steps,
            best_mission_cost,
            output_dir / "{}_last.pt".format(prefix),
        )
        eval_env.close()

    return policy
