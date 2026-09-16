"""One command-line entry point for every interception agent and mode."""

import argparse
import json

from config import ExperimentConfig
from interception_env import InterceptionEnv
from agent.a2c import train as train_a2c
from agent.ddpg import train as train_ddpg
from agent.ppo import train as train_ppo
from agent.sac import train as train_sac
from agent.td3 import train as train_td3


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["ppo", "a2c", "ddpg", "td3", "sac"], default="ppo")
    parser.add_argument("--mode", choices=["pn", "e2e"], default="pn")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--physics-engine", choices=["rotorpy", "simple"], default="rotorpy")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--additional-steps", type=int, default=None)
    parser.add_argument("--sanity", action="store_true", help="run one deterministic PN/device feasibility episode")
    return parser.parse_args()


def train_selected_agent(agent_name: str, config: ExperimentConfig):
    if agent_name == "ppo":
        return train_ppo(config)
    if agent_name == "a2c":
        return train_a2c(config)
    if agent_name == "ddpg":
        return train_ddpg(config)
    if agent_name == "td3":
        return train_td3(config)
    return train_sac(config)


def restore_saved_environment(agent_name: str, config: ExperimentConfig) -> None:
    """Resume with the exact physical environment saved by the original run."""

    config_path = config.agent_run_dir(agent_name) / "seed_{}_config.json".format(config.seed)
    if not config_path.exists():
        raise FileNotFoundError("resume configuration not found: {}".format(config_path))
    runtime_values = {
        "mode": config.mode,
        "seed": config.seed,
        "device": config.device,
        "total_train_steps": config.total_train_steps,
        "eval_interval_steps": config.eval_interval_steps,
        "save_interval_steps": config.save_interval_steps,
        "n_eval_episodes": config.n_eval_episodes,
        "overwrite": config.overwrite,
        "resume": config.resume,
        "additional_train_steps": config.additional_train_steps,
        "run_root": config.run_root,
    }
    with config_path.open(encoding="utf-8") as file:
        saved = json.load(file)
    config.load_dict(saved["environment"])
    for name, value in runtime_values.items():
        setattr(config, name, value)


def run_sanity(config: ExperimentConfig) -> None:
    env = InterceptionEnv(config)
    state, _ = env.reset(seed=config.seed)
    terminated = truncated = False
    episode_return = 0.0
    info = {}
    try:
        while not (terminated or truncated):
            state, reward, terminated, truncated, info = env.step(env.heuristic_action())
            episode_return += reward
    finally:
        env.close()
    print("reason={} success={} return={:.2f} time={:.2f}s".format(
        info["termination_reason"], info["success"], episode_return, info["episode_time"]
    ))


def main() -> None:
    args = arguments()
    config = ExperimentConfig()
    config.mode = args.mode
    config.seed = args.seed
    if args.total_steps is not None:
        config.total_train_steps = args.total_steps
    if args.eval_interval is not None:
        config.eval_interval_steps = args.eval_interval
    if args.save_interval is not None:
        config.save_interval_steps = args.save_interval
    if args.eval_episodes is not None:
        config.n_eval_episodes = args.eval_episodes
    if args.run_root is not None:
        config.run_root = args.run_root
    config.physics_engine = args.physics_engine
    config.device = args.device
    config.overwrite = args.overwrite
    config.resume = args.resume
    if args.additional_steps is not None:
        config.additional_train_steps = args.additional_steps
    if config.resume and config.additional_train_steps <= 0:
        raise ValueError("--resume requires a positive --additional-steps")
    if config.resume and config.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")
    if config.resume:
        restore_saved_environment(args.agent, config)
    config.validate()
    if args.sanity:
        run_sanity(config)
    else:
        train_selected_agent(args.agent, config)


if __name__ == "__main__":
    main()
