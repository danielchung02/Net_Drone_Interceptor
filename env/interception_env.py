"""Gymnasium environment for the PN-vs-end-to-end interception experiment.

The environment owns the generic payload state because it is episode-local and
has no meaning outside the interception task.  Both policy modes share the
same target, payload, initial-state sampler, low-level quadrotor backend, and
observation.  They differ only in where the high-level guidance acceleration
comes from.

Gymnasium semantics used here are intentional:

* capture / breach / out_of_bounds -> ``terminated=True`` (no bootstrap)
* timeout -> ``truncated=True`` (bootstrap from the returned final next state)

``timeout`` is still a mission failure for evaluation.  It is not an MDP
absorbing terminal state, however, so learner code must never replace its
returned observation with a reset observation before computing a critic target.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from config import ExperimentConfig
from env.pn import ProportionalNavigation
from env.target import FixedTargetManeuver


GRAVITY = 9.81
EPSILON = 1e-8


def _clip_vector_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm > max_norm > 0.0:
        return vector * (max_norm / norm)
    return vector


def _unit_vector(vector: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm > EPSILON:
        return vector / norm
    if fallback is None:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return np.asarray(fallback, dtype=np.float64) / np.linalg.norm(fallback)


def _sphere_entry_fraction(start: np.ndarray, end: np.ndarray, radius: float) -> Optional[float]:
    """First linear-segment fraction that enters a sphere, if any."""

    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    if np.linalg.norm(start) <= radius:
        return 0.0
    delta = end - start
    coefficient_a = float(np.dot(delta, delta))
    coefficient_b = 2.0 * float(np.dot(start, delta))
    coefficient_c = float(np.dot(start, start) - radius**2)
    discriminant = coefficient_b**2 - 4.0 * coefficient_a * coefficient_c
    if coefficient_a <= EPSILON or discriminant < 0.0:
        return None
    roots = sorted(
        [
            (-coefficient_b - np.sqrt(discriminant)) / (2.0 * coefficient_a),
            (-coefficient_b + np.sqrt(discriminant)) / (2.0 * coefficient_a),
        ]
    )
    for root in roots:
        if -EPSILON <= root <= 1.0 + EPSILON:
            return float(np.clip(root, 0.0, 1.0))
    return None


def _sphere_exit_fraction(start: np.ndarray, end: np.ndarray, radius: float) -> Optional[float]:
    """First linear-segment fraction that exits a sphere, if any."""

    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    if np.linalg.norm(start) >= radius:
        return 0.0
    delta = end - start
    coefficient_a = float(np.dot(delta, delta))
    coefficient_b = 2.0 * float(np.dot(start, delta))
    coefficient_c = float(np.dot(start, start) - radius**2)
    discriminant = coefficient_b**2 - 4.0 * coefficient_a * coefficient_c
    if coefficient_a <= EPSILON or discriminant < 0.0:
        return None
    roots = sorted(
        [
            (-coefficient_b - np.sqrt(discriminant)) / (2.0 * coefficient_a),
            (-coefficient_b + np.sqrt(discriminant)) / (2.0 * coefficient_a),
        ]
    )
    for root in roots:
        if -EPSILON <= root <= 1.0 + EPSILON:
            return float(np.clip(root, 0.0, 1.0))
    return None


def _box_exit_fraction(start: np.ndarray, end: np.ndarray, limit: float) -> Optional[float]:
    """First fraction where a segment leaves the cube ``[-limit, limit]^3``."""

    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    if np.any(np.abs(start) >= limit):
        return 0.0
    fractions = []
    delta = end - start
    for index in range(3):
        if end[index] > limit and delta[index] > EPSILON:
            fractions.append((limit - start[index]) / delta[index])
        elif end[index] < -limit and delta[index] < -EPSILON:
            fractions.append((-limit - start[index]) / delta[index])
    if not fractions:
        return None
    return float(np.clip(min(fractions), 0.0, 1.0))


class SimpleQuadrotorBackend:
    """Fast fallback used only for semantic tests and debugging.

    The thesis runs should select ``physics_backend='rotorpy'``.  This small
    backend is useful because it keeps environment/unit tests runnable without
    a graphics stack or RotorPy installation while preserving command limits.
    """

    def __init__(self, max_acceleration: float, max_speed: float, tracking_time_constant: float):
        self.max_acceleration = float(max_acceleration)
        self.max_speed = float(max_speed)
        self.tracking_time_constant = float(tracking_time_constant)
        self.position = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.actual_acceleration = np.zeros(3, dtype=np.float64)

    def reset(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self.position = np.asarray(position, dtype=np.float64).copy()
        self.velocity = np.asarray(velocity, dtype=np.float64).copy()
        self.actual_acceleration = np.zeros(3, dtype=np.float64)

    def step(self, acceleration_command: np.ndarray, dt: float) -> None:
        desired_acceleration = _clip_vector_norm(acceleration_command, self.max_acceleration)
        blend = min(1.0, dt / max(self.tracking_time_constant, dt))
        self.actual_acceleration += blend * (desired_acceleration - self.actual_acceleration)
        self.actual_acceleration = _clip_vector_norm(self.actual_acceleration, self.max_acceleration)
        self.velocity += self.actual_acceleration * dt
        self.velocity = _clip_vector_norm(self.velocity, self.max_speed)
        self.position += self.velocity * dt


class RotorPyQuadrotorBackend:
    """Single-vehicle adapter around RotorPy's physical ``Multirotor`` model.

    RotorPy's ``cmd_acc`` abstraction internally runs its attitude/controller
    and motor allocation.  The command below is a world-frame desired
    *translational* acceleration, converted to the mass-normalized thrust
    vector expected by RotorPy by adding gravity.  Thus neither PN nor E2E
    policy ever commands motor RPM/raw thrust directly.
    """

    def __init__(self, config: ExperimentConfig, max_acceleration: float, max_speed: float):
        self.config = config
        self.max_acceleration = float(max_acceleration)
        self.max_speed = float(max_speed)
        self.vehicle = None
        self.state: Dict[str, np.ndarray] = {}
        self.quad_params = self._load_quad_params(config.rotorpy_vehicle)

    @staticmethod
    def _load_quad_params(vehicle_name: str) -> Dict[str, object]:
        try:
            if vehicle_name == "crazyflie":
                from rotorpy.vehicles.crazyflie_params import quad_params
            elif vehicle_name == "hummingbird":
                from rotorpy.vehicles.hummingbird_params import quad_params
            else:
                raise ValueError("unknown RotorPy vehicle '{}'".format(vehicle_name))
        except ImportError as error:
            raise ImportError(
                "RotorPy is required for physics_backend='rotorpy'. "
                "Install the project requirements, or use physics_backend='simple' "
                "only for environment-semantic smoke tests."
            ) from error
        return quad_params

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.state["x"], dtype=np.float64)

    @property
    def velocity(self) -> np.ndarray:
        return np.asarray(self.state["v"], dtype=np.float64)

    def reset(self, position: np.ndarray, velocity: np.ndarray) -> None:
        try:
            from rotorpy.vehicles.multirotor import Multirotor
        except ImportError as error:
            raise ImportError("RotorPy Multirotor could not be imported") from error

        rotor_count = int(self.quad_params["num_rotors"])
        hover_speed = np.sqrt(
            self.quad_params["mass"] * GRAVITY / (rotor_count * self.quad_params["k_eta"])
        )
        initial_state = {
            "x": np.asarray(position, dtype=np.float64).copy(),
            "v": np.asarray(velocity, dtype=np.float64).copy(),
            "q": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            "w": np.zeros(3, dtype=np.float64),
            "wind": np.zeros(3, dtype=np.float64),
            "rotor_speeds": np.full(rotor_count, hover_speed, dtype=np.float64),
        }
        self.vehicle = Multirotor(
            self.quad_params,
            initial_state=initial_state,
            control_abstraction="cmd_acc",
            aero=self.config.rotorpy_aero,
            enable_ground=False,
        )
        self.state = initial_state

    def step(self, acceleration_command: np.ndarray, dt: float) -> None:
        if self.vehicle is None:
            raise RuntimeError("reset() must be called before step()")
        acceleration_command = _clip_vector_norm(acceleration_command, self.max_acceleration)
        thrust_acceleration = acceleration_command + np.array([0.0, 0.0, GRAVITY])
        # RotorPy normalizes this vector to construct a desired attitude.  Keep
        # an accidental exact zero from producing a numerical division by zero.
        if np.linalg.norm(thrust_acceleration) < 0.1:
            thrust_acceleration[2] = 0.1
        speed = float(np.linalg.norm(self.state["v"]))
        if speed > self.max_speed > 0.0:
            velocity_direction = self.state["v"] / speed
            outward_acceleration = max(0.0, float(np.dot(acceleration_command, velocity_direction)))
            braking_acceleration = outward_acceleration + self.config.interceptor_speed_limit_gain * (speed - self.max_speed)
            acceleration_command = _clip_vector_norm(
                acceleration_command - braking_acceleration * velocity_direction,
                self.max_acceleration,
            )
            thrust_acceleration = acceleration_command + np.array([0.0, 0.0, GRAVITY])
            if np.linalg.norm(thrust_acceleration) < 0.1:
                thrust_acceleration[2] = 0.1

        # RotorPy uses [1, 0, 0] as an intermediate desired-yaw reference.
        # A thrust vector parallel to that axis makes its cross product zero.
        # Tilt by a bounded 2.9 degrees in that singular case instead of
        # sending a NaN-producing vector to the physical simulator.
        thrust_norm = float(np.linalg.norm(thrust_acceleration))
        if thrust_norm > EPSILON and np.hypot(thrust_acceleration[1], thrust_acceleration[2]) < 0.05 * thrust_norm:
            sign = 1.0 if thrust_acceleration[1] >= 0.0 else -1.0
            thrust_acceleration[1] = sign * 0.05 * thrust_norm

        # Do not overwrite RotorPy's integrated velocity afterward.  The
        # high-level braking command above preserves a physically consistent
        # position/velocity/attitude/motor state.
        self.state = self.vehicle.step(self.state, {"cmd_acc": thrust_acceleration}, dt)


class InterceptionEnv(gym.Env):
    """One non-vectorized Gymnasium environment for a single encounter."""

    metadata = {"render_modes": ["none", "human", "rgb_array"], "render_fps": 20}
    OBSERVATION_DIM = 31

    def __init__(self, config: ExperimentConfig, render_mode: str = "none"):
        super().__init__()
        self.config = config
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(config.action_dim,),
            dtype=np.float32,
        )
        # 16 required GT features + all controllable pending/payload state.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.OBSERVATION_DIM,),
            dtype=np.float32,
        )

        self.target_maneuver = FixedTargetManeuver(
            asset_position=np.zeros(3, dtype=np.float64),
            goal_acceleration=0.0,
            goal_speed=config.target_cruise_speed,
            goal_speed_gain=config.target_goal_gain,
            max_speed=config.target_max_speed,
            max_acceleration=config.target_max_acceleration,
            max_jerk=config.target_max_jerk,
            max_climb_rate=config.target_max_climb_rate,
            goal_gain=config.target_goal_gain,
            lateral_amplitude=config.target_lateral_amplitude,
            vertical_amplitude=config.target_vertical_amplitude,
            lateral_frequency=config.target_lateral_frequency,
            vertical_frequency=config.target_vertical_frequency,
            noise_std=config.target_noise_std,
            noise_correlation_time=config.target_noise_correlation_time,
        )
        self.pn = ProportionalNavigation(
            navigation_constant=config.pn_navigation_constant,
            max_acceleration=config.interceptor_max_acceleration,
        )
        self.interceptor = self._make_backend(
            max_acceleration=config.interceptor_max_acceleration,
            max_speed=config.interceptor_max_speed,
        )
        self.target = self._make_backend(
            max_acceleration=config.target_max_acceleration,
            max_speed=config.target_max_speed,
        )

        self.asset_position = np.zeros(3, dtype=np.float64)
        self.rng = np.random.default_rng(config.seed)
        self._figure = None
        self._axis = None
        self._reset_episode_state()

    def _make_backend(self, max_acceleration: float, max_speed: float):
        if self.config.physics_backend == "rotorpy":
            return RotorPyQuadrotorBackend(self.config, max_acceleration, max_speed)
        return SimpleQuadrotorBackend(
            max_acceleration=max_acceleration,
            max_speed=max_speed,
            tracking_time_constant=self.config.interceptor_tracking_time_constant,
        )

    def _reset_episode_state(self) -> None:
        self.time = 0.0
        self.shots_used = 0
        self.cooldown_remaining = 0.0
        self.pending_launch_direction: Optional[np.ndarray] = None
        self.pending_release_remaining = 0.0
        self.payload_active = False
        self.payload_position: Optional[np.ndarray] = None
        self.payload_velocity: Optional[np.ndarray] = None
        self.payload_origin: Optional[np.ndarray] = None
        self.payload_elapsed = 0.0
        self.capture_distance: Optional[float] = None
        self.capture_time: Optional[float] = None
        self.capture_shot: Optional[int] = None
        self.control_effort = 0.0
        self.last_guidance_acceleration = np.zeros(3, dtype=np.float64)
        self.last_actual_guidance_acceleration = np.zeros(3, dtype=np.float64)
        self.last_termination_reason: Optional[str] = None
        self.termination_event_time: Optional[float] = None
        self.history = {"time": [], "target": [], "interceptor": [], "payload": []}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, object]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            episode_seed = int(seed)
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
        else:
            episode_seed = int(self.rng.integers(0, 2**31 - 1))

        self._reset_episode_state()
        interceptor_position, interceptor_velocity, target_position, target_velocity = self._sample_initial_state()
        self.interceptor.reset(interceptor_position, interceptor_velocity)
        self.target.reset(target_position, target_velocity)
        # A distinct deterministic stream keeps target noise reproducible even
        # if future initial-geometry sampling adds random draws.
        self.target_maneuver.reset(seed=episode_seed + 1)
        self._append_history()

        observation = self._get_observation()
        info = self._get_info(termination_reason=None)
        info["episode_seed"] = episode_seed
        return observation, info

    def _sample_initial_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample the required two-sphere intersection geometry exactly."""

        target_direction = _unit_vector(self.rng.normal(size=3))
        target_position = self.config.R_T * target_direction

        # Intersection of ||p_I|| = R_I and ||p_I-p_T|| = R_IT.
        distance_to_target = self.config.R_T
        center_distance = (
            self.config.R_I**2 - self.config.R_IT**2 + distance_to_target**2
        ) / (2.0 * distance_to_target)
        circle_radius_squared = max(self.config.R_I**2 - center_distance**2, 0.0)
        circle_radius = np.sqrt(circle_radius_squared)
        circle_center = center_distance * target_direction

        reference_axis = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(target_direction, reference_axis))) > 0.95:
            reference_axis = np.array([0.0, 1.0, 0.0])
        basis_u = _unit_vector(np.cross(target_direction, reference_axis))
        basis_v = _unit_vector(np.cross(target_direction, basis_u))
        angle = float(self.rng.uniform(0.0, 2.0 * np.pi))
        interceptor_position = circle_center + circle_radius * (
            np.cos(angle) * basis_u + np.sin(angle) * basis_v
        )

        random_direction = self.rng.normal(size=3)
        tangent_candidate = random_direction - (
            np.dot(random_direction, interceptor_position)
            * interceptor_position
            / max(np.dot(interceptor_position, interceptor_position), EPSILON)
        )
        patrol_direction = _unit_vector(tangent_candidate, fallback=basis_u)
        interceptor_velocity = self.config.patrol_speed * patrol_direction
        target_velocity = -self.config.target_initial_speed * target_direction
        return interceptor_position, interceptor_velocity, target_position, target_velocity

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        previous_interceptor_target_distance = float(
            np.linalg.norm(self.target.position - self.interceptor.position)
        )
        previous_target_hazard = self._hazard(float(np.linalg.norm(self.target.position)))
        previous_control_effort = self.control_effort

        guidance_acceleration, launch_direction, release_delay = self._interpret_action(action)
        self.last_guidance_acceleration = guidance_acceleration.copy()
        if self._is_launch_ready():
            self._lock_launch_plan(launch_direction, release_delay)

        termination_reason: Optional[str] = None
        for _ in range(self.config.physics_substeps):
            termination_reason = self._physics_substep(guidance_acceleration, self.config.physics_dt)
            if termination_reason is not None:
                break

        self._append_history()
        self.last_termination_reason = termination_reason
        reward = self._compute_reward(
            previous_interceptor_target_distance,
            previous_target_hazard,
            self.control_effort - previous_control_effort,
            termination_reason,
        )

        # A timeout is deliberately a time-limit truncation.  It is a failed
        # *mission* in evaluation info, while its training reward remains the
        # ordinary final-step shaping reward so critic bootstrap is coherent.
        terminated = termination_reason in {"capture", "breach", "out_of_bounds"}
        truncated = termination_reason == "timeout"
        observation = self._get_observation()
        info = self._get_info(termination_reason)
        if terminated or truncated:
            # Direct non-autoreset environments return this same observation.
            # Keeping a named copy also makes vector-wrapper integration clear.
            info["final_observation"] = observation.copy()
        if truncated:
            info["TimeLimit.truncated"] = True
        return observation, float(reward), terminated, truncated, info

    def _physics_substep(self, guidance_acceleration: np.ndarray, dt: float) -> Optional[str]:
        if self.pending_launch_direction is not None and not self.payload_active:
            self.pending_release_remaining -= dt
            if self.pending_release_remaining <= 0.0:
                self._launch_payload()

        previous_time = self.time
        previous_target_position = self.target.position.copy()
        previous_interceptor_position = self.interceptor.position.copy()
        previous_payload_position = None
        previous_payload_elapsed = self.payload_elapsed
        if self.payload_active and self.payload_position is not None:
            previous_payload_position = self.payload_position.copy()

        target_acceleration = self.target_maneuver.step(
            self.target.position,
            self.target.velocity,
            dt,
        )
        previous_interceptor_velocity = self.interceptor.velocity.copy()
        self.interceptor.step(guidance_acceleration, dt)
        self.target.step(target_acceleration, dt)
        self.time += dt
        self.last_actual_guidance_acceleration = (
            self.interceptor.velocity - previous_interceptor_velocity
        ) / dt
        self.control_effort += float(
            np.dot(self.last_actual_guidance_acceleration, self.last_actual_guidance_acceleration)
            / max(self.config.interceptor_max_acceleration**2, EPSILON)
            * dt
        )

        capture_fraction: Optional[float] = None
        payload_miss_fraction: Optional[float] = None
        if self.payload_active:
            self._update_payload(
                previous_payload_position=previous_payload_position,
                dt=dt,
            )
            capture_fraction = self._capture_fraction(
                previous_payload_position=previous_payload_position,
                previous_target_position=previous_target_position,
            )
            payload_miss_fraction = self._payload_miss_fraction(
                previous_payload_position=previous_payload_position,
                previous_payload_elapsed=previous_payload_elapsed,
                dt=dt,
            )

        # All terminal candidates are compared within the same physics
        # substep.  This prevents an end-of-step capture from incorrectly
        # overriding a breach or timeout that happened earlier in the step.
        terminal_candidates = []
        if capture_fraction is not None and (
            payload_miss_fraction is None or capture_fraction <= payload_miss_fraction + EPSILON
        ):
            terminal_candidates.append((capture_fraction, 0, "capture"))

        breach_fraction = _sphere_entry_fraction(
            previous_target_position - self.asset_position,
            self.target.position - self.asset_position,
            self.config.R_A,
        )
        if breach_fraction is not None:
            terminal_candidates.append((breach_fraction, 1, "breach"))

        out_of_bounds_fractions = [
            _box_exit_fraction(previous_interceptor_position, self.interceptor.position, self.config.airspace_limit),
            _box_exit_fraction(previous_target_position, self.target.position, self.config.airspace_limit),
        ]
        out_of_bounds_fractions = [fraction for fraction in out_of_bounds_fractions if fraction is not None]
        if out_of_bounds_fractions:
            terminal_candidates.append((min(out_of_bounds_fractions), 2, "out_of_bounds"))

        if previous_time < self.config.max_episode_time <= self.time:
            timeout_fraction = (self.config.max_episode_time - previous_time) / dt
            terminal_candidates.append((float(np.clip(timeout_fraction, 0.0, 1.0)), 3, "timeout"))

        if terminal_candidates:
            event_fraction, _, termination_reason = min(terminal_candidates, key=lambda item: (item[0], item[1]))
            self.termination_event_time = float(previous_time + event_fraction * dt)
            if termination_reason == "capture":
                self._record_capture(event_fraction, previous_target_position, previous_time, dt)
            return termination_reason

        payload_missed = payload_miss_fraction is not None
        if payload_missed:
            self._register_payload_miss()

        if not self.payload_active and self.pending_launch_direction is None and not payload_missed:
            self.cooldown_remaining = max(0.0, self.cooldown_remaining - dt)

        return None

    def _interpret_action(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        if self.config.mode == "pn":
            relative_position = self.target.position - self.interceptor.position
            relative_velocity = self.target.velocity - self.interceptor.velocity
            guidance_acceleration = self.pn(relative_position, relative_velocity)
            launch_action = action
        else:
            guidance_acceleration = action[:3].astype(np.float64) * self.config.interceptor_max_acceleration
            guidance_acceleration = _clip_vector_norm(
                guidance_acceleration,
                self.config.interceptor_max_acceleration,
            )
            launch_action = action[3:]

        azimuth = float(launch_action[0]) * np.pi
        elevation_fraction = 0.5 * (float(launch_action[1]) + 1.0)
        elevation = self.config.min_elevation + elevation_fraction * (
            self.config.max_elevation - self.config.min_elevation
        )
        release_delay = 0.5 * (float(launch_action[2]) + 1.0) * self.config.max_release_delay
        launch_direction = np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ],
            dtype=np.float64,
        )
        return guidance_acceleration, launch_direction, release_delay

    def _is_launch_ready(self) -> bool:
        return (
            not self.payload_active
            and self.pending_launch_direction is None
            and self.cooldown_remaining <= EPSILON
            and self.shots_used < self.config.maximum_shots
        )

    def _lock_launch_plan(self, launch_direction: np.ndarray, release_delay: float) -> None:
        self.pending_launch_direction = _unit_vector(launch_direction)
        self.pending_release_remaining = max(0.0, float(release_delay))
        if self.pending_release_remaining <= EPSILON:
            self._launch_payload()

    def _launch_payload(self) -> None:
        if self.pending_launch_direction is None or self.shots_used >= self.config.maximum_shots:
            return
        self.payload_active = True
        self.payload_origin = self.interceptor.position.copy()
        self.payload_position = self.interceptor.position.copy()
        self.payload_velocity = (
            self.interceptor.velocity.copy()
            + self.config.payload_speed * self.pending_launch_direction
        )
        self.payload_elapsed = 0.0
        self.shots_used += 1
        self.pending_launch_direction = None
        self.pending_release_remaining = 0.0

    def _update_payload(
        self,
        previous_payload_position: Optional[np.ndarray],
        dt: float,
    ) -> None:
        if previous_payload_position is None or self.payload_position is None or self.payload_velocity is None:
            return
        self.payload_position = self.payload_position + self.payload_velocity * dt
        self.payload_elapsed += dt

    def _capture_fraction(
        self,
        previous_payload_position: Optional[np.ndarray],
        previous_target_position: np.ndarray,
    ) -> Optional[float]:
        """First entry into the capture sphere over a physics substep."""

        if previous_payload_position is None or self.payload_position is None:
            return None
        relative_start = previous_payload_position - previous_target_position
        relative_end = self.payload_position - self.target.position
        return _sphere_entry_fraction(relative_start, relative_end, self.config.payload_capture_radius)

    def _record_capture(
        self,
        fraction: float,
        previous_target_position: np.ndarray,
        previous_time: float,
        dt: float,
    ) -> None:
        capture_position = previous_target_position + fraction * (self.target.position - previous_target_position)
        self.capture_distance = float(np.linalg.norm(capture_position - self.asset_position))
        self.capture_time = float(previous_time + fraction * dt)
        self.capture_shot = int(self.shots_used)

    def _payload_miss_fraction(
        self,
        previous_payload_position: Optional[np.ndarray],
        previous_payload_elapsed: float,
        dt: float,
    ) -> Optional[float]:
        if (
            not self.payload_active
            or previous_payload_position is None
            or self.payload_position is None
            or self.payload_origin is None
        ):
            return None
        fractions = []
        time_after = previous_payload_elapsed + dt
        if previous_payload_elapsed < self.config.payload_max_flight_time <= time_after:
            fractions.append((self.config.payload_max_flight_time - previous_payload_elapsed) / dt)
        range_fraction = _sphere_exit_fraction(
            previous_payload_position - self.payload_origin,
            self.payload_position - self.payload_origin,
            self.config.payload_max_range,
        )
        if range_fraction is not None:
            fractions.append(range_fraction)
        bounds_fraction = _box_exit_fraction(
            previous_payload_position,
            self.payload_position,
            self.config.airspace_limit,
        )
        if bounds_fraction is not None:
            fractions.append(bounds_fraction)
        if not fractions:
            return None
        return float(np.clip(min(fractions), 0.0, 1.0))

    def _register_payload_miss(self) -> None:
        self.payload_active = False
        self.payload_position = None
        self.payload_velocity = None
        self.payload_origin = None
        self.cooldown_remaining = self.config.cooldown

    def _is_miss(self) -> bool:
        if not self.payload_active or self.payload_position is None or self.payload_origin is None:
            return False
        out_of_range = np.linalg.norm(self.payload_position - self.payload_origin) > self.config.payload_max_range
        out_of_time = self.payload_elapsed >= self.config.payload_max_flight_time
        out_of_bounds = np.any(np.abs(self.payload_position) > self.config.airspace_limit)
        return bool(out_of_range or out_of_time or out_of_bounds)

    def _check_termination(self) -> Optional[str]:
        """Endpoint diagnostic helper; event ordering is handled per substep."""
        if np.linalg.norm(self.target.position - self.asset_position) <= self.config.R_A:
            return "breach"
        drone_out_of_bounds = (
            np.any(np.abs(self.interceptor.position) > self.config.airspace_limit)
            or np.any(np.abs(self.target.position) > self.config.airspace_limit)
        )
        if drone_out_of_bounds:
            return "out_of_bounds"
        if self.time >= self.config.max_episode_time:
            return "timeout"
        return None

    def _hazard(self, capture_distance: float) -> float:
        minimum_safe_distance = self.config.R_A + self.config.R_S
        if capture_distance <= minimum_safe_distance:
            return 1.0
        return float(minimum_safe_distance / capture_distance)

    def _compute_reward(
        self,
        previous_interceptor_target_distance: float,
        previous_target_hazard: float,
        control_effort_increment: float,
        termination_reason: Optional[str],
    ) -> float:
        if termination_reason == "capture":
            assert self.capture_distance is not None
            return self.config.capture_reward * (1.0 - self._hazard(self.capture_distance)) - self.config.shot_cost * self.shots_used
        if termination_reason in {"breach", "out_of_bounds"}:
            return -self.config.failure_reward - self.config.shot_cost * self.shots_used
        if not self.config.use_dense_shaping:
            return 0.0

        current_distance = float(np.linalg.norm(self.target.position - self.interceptor.position))
        current_hazard = self._hazard(float(np.linalg.norm(self.target.position - self.asset_position)))
        closing_term = self.config.closing_reward_weight * np.clip(
            previous_interceptor_target_distance - current_distance,
            -1.0,
            1.0,
        )
        hazard_term = -self.config.hazard_penalty_weight * max(0.0, current_hazard - previous_target_hazard)
        time_term = -self.config.time_penalty_weight * self.config.control_dt
        control_term = -self.config.control_penalty_weight * control_effort_increment
        return float(closing_term + hazard_term + time_term + control_term)

    def _get_observation(self) -> np.ndarray:
        target_relative_position = self.target.position - self.interceptor.position
        target_relative_velocity = self.target.velocity - self.interceptor.velocity
        payload_relative_position = np.zeros(3, dtype=np.float64)
        payload_relative_velocity = np.zeros(3, dtype=np.float64)
        payload_from_origin = np.zeros(3, dtype=np.float64)
        payload_remaining_flight_time = 0.0
        if self.payload_active and self.payload_position is not None and self.payload_velocity is not None:
            payload_relative_position = self.payload_position - self.target.position
            payload_relative_velocity = self.payload_velocity - self.target.velocity
            if self.payload_origin is not None:
                payload_from_origin = self.payload_position - self.payload_origin
            payload_remaining_flight_time = max(
                0.0,
                self.config.payload_max_flight_time - self.payload_elapsed,
            )
        pending_launch_direction = np.zeros(3, dtype=np.float64)
        if self.pending_launch_direction is not None:
            pending_launch_direction = self.pending_launch_direction

        features = np.concatenate(
            [
                target_relative_position,
                target_relative_velocity,
                self.interceptor.position - self.asset_position,
                self.interceptor.velocity,
                np.array(
                    [
                        self.config.max_episode_time - self.time,
                        self.config.maximum_shots - self.shots_used,
                        float(self.payload_active),
                        self.cooldown_remaining,
                        float(self.pending_launch_direction is not None),
                        self.pending_release_remaining,
                    ],
                    dtype=np.float64,
                ),
                payload_relative_position,
                payload_relative_velocity,
                pending_launch_direction,
                payload_from_origin,
                np.array([payload_remaining_flight_time], dtype=np.float64),
            ]
        )
        if self.config.normalize_observation:
            position_scale = max(self.config.R_T, self.config.R_I, self.config.R_IT, 1.0)
            velocity_scale = max(
                self.config.payload_speed,
                self.config.interceptor_max_speed,
                self.config.target_max_speed,
                1.0,
            )
            features = features.copy()
            features[0:3] /= position_scale
            features[3:6] /= velocity_scale
            features[6:9] /= max(self.config.R_I, 1.0)
            features[9:12] /= velocity_scale
            features[12] /= max(self.config.max_episode_time, 1.0)
            features[13] /= float(self.config.maximum_shots)
            features[15] /= max(self.config.cooldown, 1.0)
            features[17] /= max(self.config.max_release_delay, 1.0)
            features[18:21] /= max(self.config.payload_max_range, 1.0)
            features[21:24] /= velocity_scale
            features[27:30] /= max(self.config.payload_max_range, 1.0)
            features[30] /= max(self.config.payload_max_flight_time, 1.0)
        return features.astype(np.float32)

    def _mission_cost(self, termination_reason: Optional[str]) -> float:
        if termination_reason == "capture" and self.capture_distance is not None:
            return self._hazard(self.capture_distance)
        return self.config.failure_mission_cost

    def _get_info(self, termination_reason: Optional[str]) -> Dict[str, object]:
        return {
            "termination_reason": termination_reason,
            "time": float(self.termination_event_time if self.termination_event_time is not None else self.time),
            "mission_cost": float(self._mission_cost(termination_reason)),
            "capture_distance": float(self.capture_distance) if self.capture_distance is not None else float("nan"),
            "capture_time": float(self.capture_time) if self.capture_time is not None else float("nan"),
            "capture_shot": int(self.capture_shot) if self.capture_shot is not None else 0,
            "shots_used": int(self.shots_used),
            "remaining_shots": int(self.config.maximum_shots - self.shots_used),
            "payload_active": bool(self.payload_active),
            "cooldown_remaining": float(self.cooldown_remaining),
            "control_effort": float(self.control_effort),
            "guidance_acceleration": self.last_guidance_acceleration.astype(np.float32).copy(),
            "actual_guidance_acceleration": self.last_actual_guidance_acceleration.astype(np.float32).copy(),
            "target_position": self.target.position.astype(np.float32).copy(),
            "interceptor_position": self.interceptor.position.astype(np.float32).copy(),
        }

    def _append_history(self) -> None:
        self.history["time"].append(float(self.time))
        self.history["target"].append(self.target.position.copy())
        self.history["interceptor"].append(self.interceptor.position.copy())
        if self.payload_active and self.payload_position is not None:
            self.history["payload"].append(self.payload_position.copy())
        else:
            self.history["payload"].append(np.array([np.nan, np.nan, np.nan], dtype=np.float64))

    def get_trajectory(self) -> Dict[str, np.ndarray]:
        return {key: np.asarray(values, dtype=np.float64).copy() for key, values in self.history.items()}

    def render(self):
        if self.render_mode == "none":
            return None
        import matplotlib.pyplot as plt

        if self._figure is None:
            self._figure = plt.figure("InterceptionEnv")
            self._axis = self._figure.add_subplot(projection="3d")
        assert self._axis is not None
        self._axis.clear()
        limit = self.config.airspace_limit * 0.65
        self._axis.set_xlim(-limit, limit)
        self._axis.set_ylim(-limit, limit)
        self._axis.set_zlim(-limit, limit)
        self._axis.set_xlabel("x [m]")
        self._axis.set_ylabel("y [m]")
        self._axis.set_zlabel("z [m]")
        self._axis.scatter(0.0, 0.0, 0.0, c="black", marker="s", label="asset")
        self._axis.scatter(*self.target.position, c="tab:red", label="target")
        self._axis.scatter(*self.interceptor.position, c="tab:blue", label="interceptor")
        if self.payload_active and self.payload_position is not None:
            self._axis.scatter(*self.payload_position, c="tab:green", label="payload")
        self._axis.legend(loc="upper right")
        self._figure.canvas.draw()
        if self.render_mode == "human":
            plt.pause(0.001)
            return None
        rgba = np.asarray(self._figure.canvas.buffer_rgba())
        return rgba[..., :3].copy()

    def close(self) -> None:
        if self._figure is not None:
            import matplotlib.pyplot as plt

            plt.close(self._figure)
            self._figure = None
            self._axis = None
