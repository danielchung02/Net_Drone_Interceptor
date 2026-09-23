"""Small checks for the distinction between terminal and time-limit endings."""

import numpy as np
import torch
import torch.optim as optim

from agent.ppo import ActorCritic, PPOHyperParameters, update as update_ppo
from config import ExperimentConfig
from interception_env import InterceptionEnv


def make_simple_config(**overrides):
    config = ExperimentConfig()
    config.physics_engine = "simple"
    for name, value in overrides.items():
        setattr(config, name, value)
    config.validate()
    return config


def test_initial_geometry_and_dimensions():
    env = InterceptionEnv(make_simple_config())
    try:
        observation, _ = env.reset(seed=123)
        assert observation.shape == env.observation_space.shape
        assert observation.shape == (26,)
        assert env.action_space.shape == (2,)
        assert np.isclose(np.linalg.norm(env.target.position), env.config.sphere_radius)
        assert np.allclose(env.interceptor.position, np.zeros(3))

        central_angle = np.arccos(
            np.clip(
                np.dot(env.target_entry, env.target_exit) / env.config.sphere_radius**2,
                -1.0,
                1.0,
            )
        )
        assert np.isclose(np.rad2deg(central_angle), 135.0)
    finally:
        env.close()


def test_timeout_is_truncated_and_keeps_final_observation():
    env = InterceptionEnv(make_simple_config(debug_max_steps=1))
    try:
        state, _ = env.reset(seed=123)
        next_state, _, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert not terminated
        assert truncated
        assert info["termination_reason"] == "debug_guard"
        # The returned final state is a physical next state, not an automatic reset.
        assert not np.array_equal(state, next_state)
    finally:
        env.close()


def test_running_reward_has_signed_approach_progress_and_time_penalty():
    env = InterceptionEnv(make_simple_config(mode="e2e", debug_max_steps=1))
    try:
        env.reset(seed=123)
        before_distance = float(np.linalg.norm(env.target.position - env.interceptor.position))
        action = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        _, reward, terminated, truncated, _ = env.step(action)
        after_distance = float(np.linalg.norm(env.target.position - env.interceptor.position))
        expected = (
            env.config.approach_progress_scale
            * (before_distance - after_distance)
            / env.config.sphere_radius
            - env.config.time_penalty_scale * env.config.control_dt / env.config.reference_time
        )
        assert not terminated
        assert truncated
        assert np.isclose(reward, expected)
    finally:
        env.close()


def test_capture_is_a_true_termination_not_a_truncation():
    env = InterceptionEnv(make_simple_config())
    try:
        env.reset(seed=123)
        env.launch_used = True
        env.net_position = env.target.position.copy()
        env.net_velocity = env.target.velocity.copy()
        _, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert terminated
        assert not truncated
        assert info["termination_reason"] == "hit"
        assert info["success"]
        assert env.config.terminal_reward < reward <= env.config.terminal_reward + env.config.capture_time_bonus
        assert info["min_net_distance"] <= env.config.net_capture_radius
    finally:
        env.close()


def test_near_miss_bonus_and_physical_jerk_limit_remain():
    env = InterceptionEnv(make_simple_config())
    try:
        env.min_net_distance = 5.0
        expected_bonus = env.config.near_miss_bonus * (1.0 - 5.0 / env.config.near_miss_distance)
        assert np.isclose(env.near_miss_reward(), expected_bonus)
        limited = env.jerk_limited(np.array([8.0, 0.0, 0.0]), np.zeros(3), 8.0, 30.0)
        assert np.isclose(np.linalg.norm(limited), 30.0 * env.config.physics_dt)
    finally:
        env.close()


def test_failure_cannot_avoid_time_penalty_by_ending_early():
    env = InterceptionEnv(make_simple_config())
    try:
        env.time = 5.0
        accumulated_time_penalty = env.config.time_penalty_scale * env.time / env.config.reference_time
        assert np.isclose(
            accumulated_time_penalty + env.remaining_time_penalty(),
            env.config.time_penalty_scale,
        )
        assert env.config.interceptor_exit_penalty > env.config.passive_exit_penalty
    finally:
        env.close()


def test_e2e_has_three_additional_guidance_actions():
    env = InterceptionEnv(make_simple_config(mode="e2e"))
    try:
        assert env.action_space.shape == (5,)
    finally:
        env.close()


def test_pn_launch_angles_are_direct_offsets_from_line_of_sight():
    env = InterceptionEnv(make_simple_config())
    try:
        env.reset(seed=123)
        env.interceptor.position[:] = 0.0
        env.target.position[:] = np.array([10.0, 0.0, 0.0])
        _, forward, source = env.decode_action(np.array([0.0, 0.0], dtype=np.float32))
        _, horizontal, _ = env.decode_action(np.array([1.0, 0.0], dtype=np.float32))
        _, vertical, _ = env.decode_action(np.array([0.0, 1.0], dtype=np.float32))

        assert source == "rl_aim"
        assert np.allclose(forward, np.array([1.0, 0.0, 0.0]))
        assert np.isclose(np.arctan2(horizontal[1], horizontal[0]), env.config.launch_azimuth_limit)
        assert np.isclose(np.arcsin(vertical[2]), env.config.launch_elevation_limit)
    finally:
        env.close()


def test_e2e_acceleration_uses_line_of_sight_basis():
    env = InterceptionEnv(make_simple_config(mode="e2e"))
    try:
        env.reset(seed=123)
        env.interceptor.position[:] = 0.0
        env.target.position[:] = np.array([10.0, 0.0, 0.0])
        aim = np.zeros(2, dtype=np.float32)

        forward, _, _ = env.decode_action(np.concatenate([np.array([1.0, 0.0, 0.0]), aim]))
        horizontal, _, _ = env.decode_action(np.concatenate([np.array([0.0, 1.0, 0.0]), aim]))
        vertical, _, _ = env.decode_action(np.concatenate([np.array([0.0, 0.0, 1.0]), aim]))

        maximum = env.config.interceptor_max_acceleration
        assert np.allclose(forward, np.array([maximum, 0.0, 0.0]))
        assert np.allclose(horizontal, np.array([0.0, maximum, 0.0]))
        assert np.allclose(vertical, np.array([0.0, 0.0, maximum]))
    finally:
        env.close()


def test_rule_timing_launches_at_fifteen_metres_and_stops_interceptor():
    env = InterceptionEnv(make_simple_config())
    try:
        env.reset(seed=10_000)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(action)

        assert terminated
        assert not truncated
        assert info["gate_ever_open"]
        assert info["launch_used"]
        assert info["launch_distance"] <= env.config.fixed_auto_launch_distance
        assert info["gate_open_steps"] == 1
        assert info["rl_aim"]
        launch_position = env.launch_position.copy()
        assert np.allclose(env.interceptor.position, launch_position)
    finally:
        env.close()


def test_curriculum_advances_from_repeated_evaluation_success_not_steps():
    config = make_simple_config()
    qualifying_success = config.curriculum_success_threshold
    for _ in range(config.curriculum_required_evaluations - 1):
        assert not config.update_launch_curriculum(qualifying_success)
        assert config.launch_curriculum_stage == 0
    assert config.update_launch_curriculum(qualifying_success)
    assert config.launch_curriculum_stage == 1
    assert config.curriculum_success_streak == 0

    assert not config.update_launch_curriculum(qualifying_success)
    assert config.curriculum_success_streak == 0


def test_e2e_stage_zero_masks_aim_and_uses_rule_aim():
    env = InterceptionEnv(make_simple_config(mode="e2e", launch_curriculum_stage=0))
    try:
        env.reset(seed=10_000)
        assert np.allclose(env.action_mask(), np.array([1.0, 1.0, 1.0, 0.0, 0.0]))
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, source = env.decode_action(action)
        assert source == "rule_aim"
    finally:
        env.close()


def test_e2e_stage_one_activates_rl_aim_only_when_rule_timing_is_met():
    env = InterceptionEnv(make_simple_config(mode="e2e", launch_curriculum_stage=1))
    try:
        env.reset(seed=10_000)
        assert np.allclose(env.action_mask(), np.array([1.0, 1.0, 1.0, 0.0, 0.0]))
        env.interceptor.position = env.target.position - 10.0 * unit_vector(env.target.velocity)
        env.interceptor.velocity = env.target.velocity + 6.0 * unit_vector(env.target.velocity)
        assert np.allclose(env.action_mask(), np.ones(5))
        _, _, source = env.decode_action(np.zeros(5, dtype=np.float32))
        assert source == "rl_aim"
    finally:
        env.close()


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def test_ppo_log_std_remains_finite_with_sparse_action_masks():
    torch.manual_seed(0)
    hyperparameters = PPOHyperParameters()
    hyperparameters.epochs = 3
    hyperparameters.minibatch_size = 32

    for action_dim in (2, 5):
        model = ActorCritic(
            26,
            action_dim,
            32,
            hyperparameters.min_log_std,
            hyperparameters.max_log_std,
        )
        optimizer = optim.Adam(model.parameters(), lr=hyperparameters.learning_rate)
        for _ in range(10):
            states = torch.randn(128, 26)
            masks = torch.zeros(128, action_dim)
            if action_dim == 5:
                masks[:, :3] = 1.0
            masks[::32, -2:] = 1.0
            with torch.no_grad():
                _, raw_actions, old_log_probs, _, _ = model.sample_action(states, masks)
            rollout = {
                "states": states,
                "raw_actions": raw_actions,
                "action_masks": masks,
                "old_log_probs": old_log_probs,
            }
            returns = torch.randn(128)
            advantages = torch.randn(128)
            update_ppo(model, optimizer, rollout, returns, advantages, hyperparameters)

        assert torch.isfinite(model.log_std).all()
        assert torch.all(model.log_std >= hyperparameters.min_log_std)
        assert torch.all(model.log_std <= hyperparameters.max_log_std)
