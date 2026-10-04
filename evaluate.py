"""Evaluate one saved policy on a common bank of unseen scenario seeds."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from config import ExperimentConfig
from interception_env import InterceptionEnv
from record import deterministic_action, load_actor


AGENTS = ["ppo", "a2c", "ddpg", "td3", "sac"]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=AGENTS, required=True)
    parser.add_argument("--mode", choices=["pn", "e2e"], default="pn")
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--launch-distance", type=float, required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--seed-start", type=int, default=20_000)
    parser.add_argument("--num-seeds", type=int, default=200)
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--physics-engine", choices=["rotorpy", "simple"], default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def finite_mean(values):
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else float("nan")


def wilson_interval(successes: int, episodes: int):
    z = 1.959963984540054
    rate = successes / episodes
    denominator = 1.0 + z * z / episodes
    center = (rate + z * z / (2.0 * episodes)) / denominator
    margin = z * math.sqrt(
        rate * (1.0 - rate) / episodes + z * z / (4.0 * episodes * episodes)
    ) / denominator
    return center - margin, center + margin


def write_rows(path: Path, fieldnames, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = arguments()
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")

    config = ExperimentConfig()
    config.mode = args.mode
    config.run_root = args.run_root
    config.fixed_auto_launch_distance = args.launch_distance
    run_dir = config.agent_run_dir(args.agent)
    prefix = "seed_{}".format(args.training_seed)

    config_path = run_dir / "{}_config.json".format(prefix)
    checkpoint_path = run_dir / "{}_{}.pt".format(prefix, args.checkpoint)
    with config_path.open(encoding="utf-8") as file:
        saved = json.load(file)
    config.load_dict(saved["environment"])
    if args.physics_engine is not None:
        config.physics_engine = args.physics_engine

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config.launch_curriculum_stage = int(
        checkpoint.get("launch_curriculum_stage", config.launch_curriculum_stage)
    )
    config.curriculum_stage_start_step = int(
        checkpoint.get(
            "curriculum_stage_start_step",
            checkpoint.get("total_steps", checkpoint.get("steps", 0)),
        )
    )
    actor = load_actor(
        args.agent,
        saved,
        checkpoint,
        InterceptionEnv.observation_dim,
        config.action_dim,
        device,
    )

    episode_rows = []
    env = InterceptionEnv(config)
    try:
        for scenario_seed in range(args.seed_start, args.seed_start + args.num_seeds):
            state, _ = env.reset(seed=scenario_seed)
            terminated = False
            truncated = False
            episode_return = 0.0
            info = {}
            while not (terminated or truncated):
                action = deterministic_action(args.agent, actor, state, device)
                state, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)

            episode_rows.append(
                {
                    "scenario_seed": scenario_seed,
                    "success": int(bool(info["success"])),
                    "episode_return": episode_return,
                    "termination_reason": info["termination_reason"],
                    "episode_time": info["episode_time"],
                    "capture_time": info["capture_time"],
                    "min_distance": info["min_distance"],
                    "min_net_distance": info["min_net_distance"],
                    "control_effort": info["control_effort"],
                    "launch_distance": info["launch_distance"],
                    "launch_time": info["launch_time"],
                    "launch_used": int(bool(info["launch_used"])),
                    "gate_ever_open": int(bool(info["gate_ever_open"])),
                }
            )
            completed = len(episode_rows)
            if completed % 25 == 0 or completed == args.num_seeds:
                print(
                    "distance={:g}m agent={} evaluated {}/{} seeds".format(
                        args.launch_distance, args.agent, completed, args.num_seeds
                    ),
                    flush=True,
                )
    finally:
        env.close()

    successes = sum(row["success"] for row in episode_rows)
    success_rate = successes / args.num_seeds
    ci_low, ci_high = wilson_interval(successes, args.num_seeds)
    summary = {
        "training_seed": args.training_seed,
        "checkpoint": args.checkpoint,
        "launch_distance": args.launch_distance,
        "test_seed_start": args.seed_start,
        "test_seed_end": args.seed_start + args.num_seeds - 1,
        "num_test_seeds": args.num_seeds,
        "successes": successes,
        "success_rate": success_rate,
        "wilson_95_low": ci_low,
        "wilson_95_high": ci_high,
        "mean_return": float(np.mean([row["episode_return"] for row in episode_rows])),
        "mean_capture_time": finite_mean([row["capture_time"] for row in episode_rows]),
        "mean_min_distance": finite_mean([row["min_distance"] for row in episode_rows]),
        "mean_min_net_distance": finite_mean([row["min_net_distance"] for row in episode_rows]),
        "mean_launch_distance": finite_mean([row["launch_distance"] for row in episode_rows]),
        "gate_open_rate": float(np.mean([row["gate_ever_open"] for row in episode_rows])),
    }

    episode_path = run_dir / "{}_{}_test_episodes.csv".format(prefix, args.checkpoint)
    summary_path = run_dir / "{}_{}_test_summary.csv".format(prefix, args.checkpoint)
    write_rows(episode_path, list(episode_rows[0]), episode_rows)
    write_rows(summary_path, list(summary), [summary])
    print(
        "distance={:g}m agent={} success={}/{} ({:.1%}) Wilson95%=[{:.1%}, {:.1%}]".format(
            args.launch_distance,
            args.agent,
            successes,
            args.num_seeds,
            success_rate,
            ci_low,
            ci_high,
        )
    )


if __name__ == "__main__":
    main()
