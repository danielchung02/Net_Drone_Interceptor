"""PN + fixed launch-rule feasibility check before RL training.

This is not one of the 12 learned methods.  It exists to answer the first
research sanity question: under the selected physical/reward parameters, can a
non-learning PN interceptor ever capture a non-learning target at all?
"""

import argparse

import numpy as np

from agent.common import evaluate_episodes
from config import make_config
from env.interception_env import InterceptionEnv


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the PN + fixed-launch sanity baseline.")
    parser.add_argument("--backend", choices=["rotorpy", "simple"], default="rotorpy")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--release-delay", type=float, default=0.0)
    return parser.parse_args()


def fixed_launch_action(env: InterceptionEnv, release_delay: float) -> np.ndarray:
    """Aim the generic payload at the current target and use a fixed delay."""

    line_of_sight = env.target.position - env.interceptor.position
    direction = line_of_sight / max(np.linalg.norm(line_of_sight), 1e-8)
    azimuth = np.arctan2(direction[1], direction[0])
    elevation = np.arcsin(np.clip(direction[2], -1.0, 1.0))
    elevation = np.clip(elevation, env.config.min_elevation, env.config.max_elevation)
    normalized_elevation = 2.0 * (elevation - env.config.min_elevation) / (
        env.config.max_elevation - env.config.min_elevation
    ) - 1.0
    normalized_delay = 2.0 * np.clip(release_delay, 0.0, env.config.max_release_delay) / (
        env.config.max_release_delay
    ) - 1.0
    return np.asarray(
        [azimuth / np.pi, normalized_elevation, normalized_delay],
        dtype=np.float32,
    )


def main():
    arguments = parse_arguments()
    seed_bank = tuple(range(arguments.seed_start, arguments.seed_start + arguments.episodes))
    config = make_config(
        algorithm="ppo",  # only required by config validation; no PPO is trained here
        mode="pn",
        physics_backend=arguments.backend,
        n_eval_episodes=arguments.episodes,
        eval_seed_bank=seed_bank,
    )
    env = InterceptionEnv(config=config, render_mode="none")
    try:
        metrics = evaluate_episodes(
            env,
            lambda _: fixed_launch_action(env, arguments.release_delay),
            config,
        )
    finally:
        env.close()
    for name, value in metrics.items():
        print("{}: {}".format(name, value))


if __name__ == "__main__":
    main()
