"""Evaluate PN guidance and rule-timed launch with analytic ballistic aim."""

import argparse

from agent.common import evaluate_episodes
from config import ExperimentConfig
from interception_env import InterceptionEnv


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--physics-engine", choices=["rotorpy", "simple"], default="rotorpy")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=10_000)
    return parser.parse_args()


def main():
    args = arguments()
    config = ExperimentConfig()
    config.mode = "pn"
    config.launch_curriculum_stage = 1
    config.physics_engine = args.physics_engine
    config.n_eval_episodes = args.episodes
    config.eval_seed_bank = tuple(range(args.seed_start, args.seed_start + args.episodes))
    config.validate()
    env = InterceptionEnv(config)
    try:
        metrics = evaluate_episodes(env, lambda _: env.heuristic_action(), config)
    finally:
        env.close()
    for name, value in metrics.items():
        print("{}: {}".format(name, value))


if __name__ == "__main__":
    main()
