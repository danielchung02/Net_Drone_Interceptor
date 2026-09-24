"""Record a deterministic rollout for any trained interception agent."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config import ExperimentConfig
from interception_env import InterceptionEnv
from agent.a2c import ActorCritic as A2CActorCritic
from agent.a2c import AgentHyperParameters as A2CHyperParameters
from agent.ddpg import Actor as DDPGActor
from agent.ddpg import AgentHyperParameters as DDPGHyperParameters
from agent.ppo import ActorCritic as PPOActorCritic
from agent.ppo import PPOHyperParameters
from agent.sac import AgentHyperParameters as SACHyperParameters
from agent.sac import GaussianActor as SACActor
from agent.td3 import Actor as TD3Actor
from agent.td3 import AgentHyperParameters as TD3HyperParameters


def restore_hyperparameters(saved, agent_name: str, hyperparameters) -> None:
    for name, value in saved[agent_name].items():
        if hasattr(hyperparameters, name):
            setattr(hyperparameters, name, value)


def load_actor(agent_name: str, saved, checkpoint, observation_dim: int, action_dim: int, device: torch.device):
    if agent_name == "ppo":
        hyperparameters = PPOHyperParameters()
        restore_hyperparameters(saved, agent_name, hyperparameters)
        actor = PPOActorCritic(
            observation_dim,
            action_dim,
            hyperparameters.hidden_dim,
            hyperparameters.min_log_std,
            hyperparameters.max_log_std,
        )
        actor.load_state_dict(checkpoint["model"])
    elif agent_name == "a2c":
        hyperparameters = A2CHyperParameters()
        restore_hyperparameters(saved, agent_name, hyperparameters)
        actor = A2CActorCritic(
            observation_dim,
            action_dim,
            hyperparameters.hidden_dim,
            hyperparameters.min_log_std,
            hyperparameters.max_log_std,
        )
        actor.load_state_dict(checkpoint["model_state_dict"])
    elif agent_name == "ddpg":
        hyperparameters = DDPGHyperParameters()
        restore_hyperparameters(saved, agent_name, hyperparameters)
        actor = DDPGActor(observation_dim, action_dim, hyperparameters.hidden_dim)
        actor.load_state_dict(checkpoint["actor_state_dict"])
    elif agent_name == "td3":
        hyperparameters = TD3HyperParameters()
        restore_hyperparameters(saved, agent_name, hyperparameters)
        actor = TD3Actor(observation_dim, action_dim, hyperparameters.hidden_dim)
        actor.load_state_dict(checkpoint["actor_state_dict"])
    else:
        hyperparameters = SACHyperParameters()
        restore_hyperparameters(saved, agent_name, hyperparameters)
        actor = SACActor(observation_dim, action_dim, hyperparameters.hidden_dim)
        actor.load_state_dict(checkpoint["actor_state_dict"])
    return actor.to(device).eval()


def deterministic_action(agent_name: str, actor, state: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        if agent_name in {"ddpg", "td3"}:
            action = actor(tensor)
        else:
            action = actor.deterministic_action(tensor)
    return action.squeeze(0).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["ppo", "a2c", "ddpg", "td3", "sac"], default="ppo")
    parser.add_argument("--mode", choices=["pn", "e2e"], default="pn")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario-seed", type=int, default=10_000)
    parser.add_argument(
        "--checkpoint",
        choices=["best", "last", "stage0_best", "stage1_best", "stage2_best"],
        default="best",
    )
    parser.add_argument("--physics-engine", choices=["rotorpy", "simple"], default=None)
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_root) / args.mode / args.agent
    prefix = "seed_{}".format(args.seed)
    with (run_dir / "{}_config.json".format(prefix)).open(encoding="utf-8") as file:
        saved = json.load(file)
    environment = saved["environment"]
    config = ExperimentConfig()
    config.load_dict(environment)
    if args.physics_engine is not None:
        config.physics_engine = args.physics_engine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = run_dir / "{}_{}.pt".format(prefix, args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config.launch_curriculum_stage = int(
        checkpoint.get("launch_curriculum_stage", config.launch_curriculum_stage)
    )
    config.curriculum_success_streak = int(
        checkpoint.get("curriculum_success_streak", 0)
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

    env = InterceptionEnv(config)
    state, _ = env.reset(seed=args.scenario_seed)
    terminated = truncated = False
    info = {}
    try:
        while not (terminated or truncated):
            action = deterministic_action(args.agent, actor, state, device)
            state, _, terminated, truncated, info = env.step(action)
        trajectory = env.trajectory()
    finally:
        env.close()
    rollout_name = "{}_{}_rollout_{}".format(prefix, args.checkpoint, args.scenario_seed)
    np.savez(run_dir / "{}.npz".format(rollout_name), **trajectory)
    print("reason={} success={}".format(info["termination_reason"], info["success"]))
    if not args.no_video:
        save_video(trajectory, config, run_dir / "{}.mp4".format(rollout_name))


def save_video(trajectory, config: ExperimentConfig, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    if not FFMpegWriter.isAvailable():
        raise RuntimeError("NPZ was saved, but ffmpeg is required for MP4")
    figure = plt.figure(figsize=(7, 6))
    axis = figure.add_subplot(projection="3d")
    radius = config.sphere_radius
    axis.set(xlim=(-radius, radius), ylim=(-radius, radius), zlim=(-radius, radius), xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    axis.set_box_aspect((1.0, 1.0, 1.0))
    azimuths = np.linspace(0.0, 2.0 * np.pi, 37)
    elevations = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 19)
    azimuth_grid, elevation_grid = np.meshgrid(azimuths, elevations)
    sphere_x = radius * np.cos(elevation_grid) * np.cos(azimuth_grid)
    sphere_y = radius * np.cos(elevation_grid) * np.sin(azimuth_grid)
    sphere_z = radius * np.sin(elevation_grid)
    axis.plot_wireframe(sphere_x, sphere_y, sphere_z, color="gray", alpha=0.25, linewidth=0.35)
    target_line, = axis.plot([], [], [], "r-", label="target")
    interceptor_line, = axis.plot([], [], [], "b-", label="interceptor")
    net_line, = axis.plot([], [], [], "g-", label="net")
    axis.legend()

    def frame(index):
        for line, name in ((target_line, "target"), (interceptor_line, "interceptor"), (net_line, "net")):
            points = trajectory[name][: index + 1]
            line.set_data(points[:, 0], points[:, 1])
            line.set_3d_properties(points[:, 2])
        return target_line, interceptor_line, net_line

    animation = FuncAnimation(figure, frame, frames=len(trajectory["time"]), interval=1000 / 20)
    animation.save(path, writer=FFMpegWriter(fps=20))
    plt.close(figure)


if __name__ == "__main__":
    main()
