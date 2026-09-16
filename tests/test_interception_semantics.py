"""Small checks for the distinction between terminal and time-limit endings."""

import numpy as np

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
        assert observation.shape == (17,)
        assert env.action_space.shape == (3,)
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


def test_running_reward_has_progress_and_strong_time_penalty_only():
    env = InterceptionEnv(make_simple_config(mode="e2e", debug_max_steps=1))
    try:
        env.reset(seed=123)
        before_distance = np.linalg.norm(env.target.position - env.interceptor.position)
        action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        _, reward, terminated, truncated, _ = env.step(action)
        after_distance = np.linalg.norm(env.target.position - env.interceptor.position)
        expected = 0.5 * (before_distance - after_distance) / env.config.sphere_radius
        expected -= env.config.time_penalty_scale * env.config.control_dt / env.config.reference_time
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
        assert np.isclose(env.near_miss_reward(), 2.0)
        limited = env.jerk_limited(np.array([8.0, 0.0, 0.0]), np.zeros(3), 8.0, 30.0)
        assert np.isclose(np.linalg.norm(limited), 30.0 * env.config.physics_dt)
    finally:
        env.close()


def test_e2e_has_three_additional_guidance_actions():
    env = InterceptionEnv(make_simple_config(mode="e2e"))
    try:
        assert env.action_space.shape == (6,)
    finally:
        env.close()
