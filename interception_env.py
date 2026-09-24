"""One-file RotorPy interception environment: target, PN, ballistic net, and Gym API."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import ExperimentConfig


GRAVITY = 9.81
EPS = 1e-8

# 크기 1로 정규화
def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > EPS:
        return np.asarray(vector, dtype=np.float64) / norm
    return np.array([1.0, 0.0, 0.0])

# max넘으면 max크기로 정규화
def clip_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > maximum:
        return np.asarray(vector, dtype=np.float64) * maximum / norm
    return np.asarray(vector, dtype=np.float64)


def segment_distance(start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    denominator = float(np.dot(delta, delta))
    if denominator < EPS:
        fraction = 0.0 
    else:
        fraction = float(np.clip(-np.dot(start, delta) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(start + fraction * delta))


def is_hit(start: np.ndarray, end: np.ndarray, radius: float) -> bool:
    return segment_distance(start, end) <= radius


class SimpleQuadrotor:
    def __init__(self, max_acceleration: float, max_speed: float):
        self.max_acceleration = max_acceleration
        self.max_speed = max_speed
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        self.angular_velocity = np.zeros(3)

    def reset(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self.position = np.asarray(position, dtype=np.float64).copy()
        self.velocity = np.asarray(velocity, dtype=np.float64).copy()
        self.angular_velocity[:] = 0.0

    def step(self, acceleration: np.ndarray, dt: float) -> None:
        acceleration = clip_norm(acceleration, self.max_acceleration)
        self.velocity = clip_norm(self.velocity + acceleration * dt, self.max_speed)
        self.position += self.velocity * dt


class RotorPyQuadrotor:
    """Small adapter: RL/PN command acceleration, RotorPy commands motors."""

    def __init__(self, config: ExperimentConfig, max_acceleration: float, max_speed: float, speed_limit_gain: float):
        self.config = config
        self.max_acceleration = max_acceleration
        self.max_speed = max_speed
        self.speed_limit_gain = speed_limit_gain
        self.vehicle = None
        self.state: Dict[str, np.ndarray] = {}
        try:
            if config.rotorpy_vehicle == "crazyflie":
                from rotorpy.vehicles.crazyflie_params import quad_params
            elif config.rotorpy_vehicle == "hummingbird":
                from rotorpy.vehicles.hummingbird_params import quad_params
            else:
                raise ValueError("unknown RotorPy vehicle '{}'".format(config.rotorpy_vehicle))
        except ImportError as error:
            raise ImportError("install rotorpy or use --physics-engine simple only for debugging") from error
        self.quad_params = quad_params

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.state["x"], dtype=np.float64)

    @property
    def velocity(self) -> np.ndarray:
        return np.asarray(self.state["v"], dtype=np.float64)

    @property
    def quaternion(self) -> np.ndarray:
        return np.asarray(self.state["q"], dtype=np.float64)

    @property
    def angular_velocity(self) -> np.ndarray:
        return np.asarray(self.state["w"], dtype=np.float64)

    def reset(self, position: np.ndarray, velocity: np.ndarray) -> None:
        from rotorpy.vehicles.multirotor import Multirotor

        rotor_count = int(self.quad_params["num_rotors"]) #로터 개수
        # n k w^2 = mg
        hover_rpm = np.sqrt(self.quad_params["mass"] * GRAVITY / (rotor_count * self.quad_params["k_eta"]))
        self.state = {
            "x": np.asarray(position, dtype=np.float64).copy(),
            "v": np.asarray(velocity, dtype=np.float64).copy(),
            "q": np.array([0.0, 0.0, 0.0, 1.0]),
            "w": np.zeros(3),
            "wind": np.zeros(3),
            "rotor_speeds": np.full(rotor_count, hover_rpm),
        }
        self.vehicle = Multirotor(
            self.quad_params,
            initial_state=self.state,
            control_abstraction="cmd_acc", #agent가 직접 저수준 로터 제어를 하지 않아도 되게(3차원 가속도 명령만 출력하면 됨)
            aero=self.config.rotorpy_aero,
            enable_ground=False,
        )

    def step(self, acceleration: np.ndarray, dt: float) -> None:
        if self.vehicle is None:
            raise RuntimeError("call reset before step")
        acceleration = clip_norm(acceleration, self.max_acceleration)
        speed = float(np.linalg.norm(self.velocity))
        if speed > self.max_speed:
            direction = unit(self.velocity)
            outward = max(0.0, float(np.dot(acceleration, direction)))
            brake = outward + self.speed_limit_gain * (speed - self.max_speed)
            acceleration = clip_norm(acceleration - brake * direction, self.max_acceleration)
            #speed는 clip_norm안 하고 ouward and brake를 도입한 이유는 위치 속도 자세 각속도 등을
            #  다 적분한 후에 v만 강제로 자르면 물리적으로 불일치->그래서 진행 방향 반대로 가속도 명령을 내리게끔

        # RotorPy's cmd_acc is a specific-force vector, hence +g. Its fixed
        
        command = acceleration + np.array([0.0, 0.0, GRAVITY])
        command_norm = float(np.linalg.norm(command))
        if command_norm < 0.1: # 최대 가속도가 8이므로 command = adesired +[0,0,g]에서 이 if문은 실행되지 않음. 방어적 코드일 뿐
            command[2] = 0.1 #cmd가 우연히 크기가 0이되어버리면 z축 cmd에 작은값을 넣겠다.
        elif np.hypot(command[1], command[2]) < 0.05 * command_norm: #hypot = (y^2+z^2)^0.5
            command[1] = 0.05 * command_norm # yaw reference is singular when this vector is exactly world-x.
        self.state = self.vehicle.step(self.state, {"cmd_acc": command}, dt)


class InterceptionEnv(gym.Env):
    """One encounter inside an empty sphere; no official timeout exists."""

    #metadata = {"render_modes": ["none"], "render_fps": 20} #rendering 안할거면 없어도 됨
    observation_dim = 26 #상대위치(3) 상대속도(3) 요격기위치(3) 요격기속도(3) 요격기자세쿼터니언(4) 요격기각속도(3) 직전가속도(3) 시간(1) 넷발사여부(1) gate(1) curriculum(1)

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.action_space = spaces.Box(-1.0, 1.0, shape=(config.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.observation_dim,), dtype=np.float32)
        self.rng = np.random.default_rng(config.seed) #random generator
        self.interceptor = self.make_quadrotor(
            config.interceptor_max_acceleration,
            config.interceptor_max_speed,
            config.interceptor_speed_limit_gain,
        )
        self.target = self.make_quadrotor(
            config.target_max_acceleration,
            config.target_max_speed,
            config.target_speed_limit_gain,
        )
        self.clear_episode()

    def make_quadrotor(self, max_acceleration: float, max_speed: float, speed_limit_gain: float):
        if self.config.physics_engine == "rotorpy":
            return RotorPyQuadrotor(self.config, max_acceleration, max_speed, speed_limit_gain)
        return SimpleQuadrotor(max_acceleration, max_speed)

    def clear_episode(self) -> None:
        self.time = 0.0 #현재 episode의 물리 시간. physics substep마다 physics_dt=0.01씩 증가
        self.step_count = 0 #agent가 env.step(action)을 몇 번 호출했는지
        self.launch_used = False
        self.launch_position: Optional[np.ndarray] = None #넷발사 순간의 위치
        self.net_position: Optional[np.ndarray] = None #현재 넷위치
        self.net_velocity: Optional[np.ndarray] = None #현재 넷 속도
        self.last_interceptor_acceleration = np.zeros(3) #직전 physics substep에서 적용된 요격기 가속도
        self.last_target_acceleration = np.zeros(3) #다음 desired acc와의 차이를 계산하여 jerk를 제한
        self.ou_noise = np.zeros(3)
        self.min_distance = float("inf") #episode 동안 기록한 요격기와 타겟 간 최소거리
        self.min_net_distance = float("inf") #episode 동안 기록한 그물과 타겟 간 최소거리
        self.control_effort = 0.0 #sum(요격기의 accelerator^2 * dt)
        self.launch_distance = float("nan") #발사 시점의 요격기-타겟 거리
        self.launch_time = float("nan")
        self.launch_source: Optional[str] = None
        self.launch_teacher_aim = np.full(2, np.nan, dtype=np.float32)
        self.capture_time = float("nan")
        self.gate_ever_open = False
        self.gate_open_steps = 0
        self.first_gate_time = float("nan")
        self.reason: Optional[str] = None #종료 사유(hit, miss, target_exit interceptor_exit, debug_guard)
        self.history = {"time": [], "target": [], "interceptor": [], "net": []}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, object]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.action_space.seed(seed)
        self.clear_episode()
        entry_direction = unit(self.rng.normal(size=3)) #entry를 가리키는 길이 1짜리 벡터
        exit_direction = self.sample_exit_direction(entry_direction) #실제 exit의 방향이 아니라 baseline을 이었을때 반대편 공역
        self.target_entry = self.config.sphere_radius * entry_direction
        self.target_exit = self.config.sphere_radius * exit_direction
        self.route_direction = unit(self.target_exit - self.target_entry)
        self.route_horizontal_axis, self.route_vertical_axis = self.route_basis(self.route_direction)
        self.horizontal_amplitude = float(self.rng.uniform(*self.config.target_sine_horizontal_amplitude_range)) #*은 튜플을 함수인자로 받는 문법
        self.vertical_amplitude = float(self.rng.uniform(*self.config.target_sine_vertical_amplitude_range))
        self.horizontal_frequency = float(self.rng.uniform(*self.config.target_sine_horizontal_frequency_range))
        self.vertical_frequency = float(self.rng.uniform(*self.config.target_sine_vertical_frequency_range))
        self.target.reset(self.target_entry, self.config.target_speed * self.route_direction)
        self.interceptor.reset(np.zeros(3), np.zeros(3))
        self.append_history()
        return self.observation(), self.info()

    def sample_exit_direction(self, entry_direction: np.ndarray) -> np.ndarray:
        tangent = self.rng.normal(size=3)
        tangent = tangent - np.dot(tangent, entry_direction) * entry_direction
        tangent = unit(tangent)
        angle = np.deg2rad(self.config.target_route_central_angle_degrees)
        # Entry is random and tangent is random, but every nominal route
        # subtends the same 135-degree central angle through the sphere.
        return np.cos(angle) * entry_direction + np.sin(angle) * tangent

    @staticmethod
    def route_basis(direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        reference = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(reference, direction))) > 0.9: # direction이 거의 z축에 평행하면 reference랑 거의 평행해서 horizontal을 정의하기 어렵 
            reference = np.array([0.0, 1.0, 0.0])
        horizontal_axis = unit(np.cross(direction, reference))
        vertical_axis = unit(np.cross(direction, horizontal_axis))
        return horizontal_axis, vertical_axis

    def line_of_sight_basis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        line_of_sight = unit(self.target.position - self.interceptor.position)
        reference_up = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(reference_up, line_of_sight))) > 0.9:
            reference_up = np.array([0.0, 1.0, 0.0])
        horizontal_axis = unit(np.cross(reference_up, line_of_sight))
        vertical_axis = unit(np.cross(line_of_sight, horizontal_axis))
        return line_of_sight, horizontal_axis, vertical_axis

    def ballistic_launch_direction(self) -> np.ndarray:
        """Constant-velocity ballistic lead used directly as the launch direction."""

        relative_position = self.target.position - self.interceptor.position
        relative_velocity = self.target.velocity - self.interceptor.velocity
        gravity = np.array([0.0, 0.0, -GRAVITY])
        flight_time = max(float(np.linalg.norm(relative_position)) / self.config.net_speed, 0.01)
        required_displacement = relative_position.copy()
        for _ in range(8):
            required_displacement = (
                relative_position
                + relative_velocity * flight_time
                - 0.5 * gravity * flight_time**2
            )
            updated_time = float(np.linalg.norm(required_displacement)) / self.config.net_speed
            flight_time = 0.5 * flight_time + 0.5 * max(updated_time, 0.01)
        return unit(required_displacement)

    def engagement_values(self) -> Tuple[float, float]:
        relative_position = self.target.position - self.interceptor.position
        relative_velocity = self.target.velocity - self.interceptor.velocity
        distance = max(float(np.linalg.norm(relative_position)), EPS)
        line_of_sight = relative_position / distance
        closing_speed = -float(np.dot(relative_velocity, line_of_sight))
        return distance, closing_speed

    def launch_gate_open(self) -> bool:
        """Rule-based launch timing, evaluated once at each control step."""

        distance, closing_speed = self.engagement_values()
        if distance > self.config.fixed_auto_launch_distance:
            return False
        if closing_speed < self.config.launch_gate_min_closing_speed:
            return False
        return True

    def uses_rl_aim(self) -> bool:
        """PN always learns aim; E2E learns it only after curriculum stage 0."""

        return self.config.mode == "pn" or self.config.launch_curriculum_stage >= 1

    def action_mask(self) -> np.ndarray:
        """Mark action dimensions that can affect the current transition."""

        gate_open = self.launch_gate_open()
        aim_active = float(self.uses_rl_aim() and gate_open)
        if self.config.mode == "pn":
            return np.array([aim_active, aim_active], dtype=np.float32)
        if gate_open:
            return np.array([0.0, 0.0, 0.0, aim_active, aim_active], dtype=np.float32)
        if self.config.launch_curriculum_stage == 1:
            # Guidance is executed deterministically but frozen during aim learning.
            return np.zeros(5, dtype=np.float32)
        return np.array([1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)

    def direction_to_aim_action(self, direction: np.ndarray) -> np.ndarray:
        """Convert a world-frame direction into normalized LOS-relative angles."""

        line_of_sight, horizontal_axis, vertical_axis = self.line_of_sight_basis()
        direction = unit(direction)
        elevation = np.arcsin(np.clip(np.dot(direction, vertical_axis), -1.0, 1.0))
        azimuth = np.arctan2(
            np.dot(direction, horizontal_axis),
            np.dot(direction, line_of_sight),
        )
        return np.array(
            [
                np.clip(azimuth / self.config.launch_azimuth_limit, -1.0, 1.0),
                np.clip(elevation / self.config.launch_elevation_limit, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

    def ballistic_aim_action(self) -> np.ndarray:
        return self.direction_to_aim_action(self.ballistic_launch_direction())

    def decode_action(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        line_of_sight, horizontal_axis, vertical_axis = self.line_of_sight_basis()
        if self.config.mode == "pn":
            desired = self.pn_acceleration()
            aim_values = action[:2]
        else:
            acceleration_values = action[:3].astype(np.float64)
            desired = self.config.interceptor_max_acceleration * (
                acceleration_values[0] * line_of_sight
                + acceleration_values[1] * horizontal_axis
                + acceleration_values[2] * vertical_axis
            )
            desired = clip_norm(desired, self.config.interceptor_max_acceleration)
            aim_values = action[3:5]

        if not self.uses_rl_aim():
            return desired, self.ballistic_launch_direction(), "rule_aim"

        azimuth = self.config.launch_azimuth_limit * float(aim_values[0])
        elevation = self.config.launch_elevation_limit * float(aim_values[1])
        launch_direction = (
            np.cos(elevation) * np.cos(azimuth) * line_of_sight
            + np.cos(elevation) * np.sin(azimuth) * horizontal_axis
            + np.sin(elevation) * vertical_axis
        )
        return desired, unit(launch_direction), "rl_aim"

    def pn_acceleration(self) -> np.ndarray:
        relative_position = self.target.position - self.interceptor.position
        relative_velocity = self.target.velocity - self.interceptor.velocity
        distance = max(float(np.linalg.norm(relative_position)), EPS)
        line_of_sight = relative_position / distance
        # The interceptor and target use the same 8 m/s nominal flight speed.
        # interceptor_max_speed remains only a physical upper bound.
        reference_velocity = self.config.target_speed * line_of_sight
        speed_acceleration = self.config.pn_speed_gain * (
            reference_velocity - self.interceptor.velocity
        )
        closing_speed = max(0.0, -float(np.dot(relative_velocity, line_of_sight)))
        line_of_sight_rate = np.cross(relative_position, relative_velocity) / distance**2
        lateral_acceleration = (
            self.config.pn_navigation_constant
            * closing_speed
            * np.cross(line_of_sight_rate, line_of_sight)
        )
        command = speed_acceleration + lateral_acceleration
        return clip_norm(command, self.config.interceptor_max_acceleration)

    def launch(self, direction: np.ndarray, source: str) -> None:
        self.launch_used = True
        self.launch_source = source
        self.launch_position = self.interceptor.position.copy()
        self.net_position = self.launch_position.copy()
        self.net_velocity = self.interceptor.velocity.copy() + self.config.net_speed * unit(direction)
        self.launch_distance = float(np.linalg.norm(self.target.position - self.interceptor.position))
        self.launch_time = self.time

    def resolve_launched_net(self) -> str:
        """After launch, stop the interceptor and finish the ballistic outcome internally."""

        substeps_since_history = 0
        while True:
            before_target = self.target.position.copy()
            before_net = self.net_position.copy()
            target_desired = self.target_acceleration()
            target_acceleration = self.jerk_limited(
                target_desired,
                self.last_target_acceleration,
                self.config.target_max_acceleration,
                self.config.target_max_jerk,
            )
            self.target.step(target_acceleration, self.config.physics_dt)
            self.last_target_acceleration = target_acceleration
            self.net_velocity += np.array([0.0, 0.0, -GRAVITY]) * self.config.physics_dt
            self.net_position += self.net_velocity * self.config.physics_dt
            self.time += self.config.physics_dt
            substeps_since_history += 1

            relative_start = before_net - before_target
            relative_end = self.net_position - self.target.position
            self.min_net_distance = min(
                self.min_net_distance,
                segment_distance(relative_start, relative_end),
            )
            reason = None
            if is_hit(relative_start, relative_end, self.config.net_capture_radius):
                self.capture_time = self.time
                reason = "hit"
            elif np.linalg.norm(self.net_position) > self.config.sphere_radius:
                reason = "miss"
            elif np.linalg.norm(self.target.position) > self.config.sphere_radius:
                reason = "miss"

            if substeps_since_history >= self.config.physics_substeps or reason is not None:
                self.append_history()
                substeps_since_history = 0
            if reason is not None:
                return reason

    def physics_step(self, desired_interceptor_acceleration: np.ndarray) -> Optional[str]:
        interceptor_acceleration = self.jerk_limited(
            desired_interceptor_acceleration,
            self.last_interceptor_acceleration,
            self.config.interceptor_max_acceleration,
            self.config.interceptor_max_jerk,
        )
        target_desired = self.target_acceleration()
        target_acceleration = self.jerk_limited(
            target_desired,
            self.last_target_acceleration,
            self.config.target_max_acceleration,
            self.config.target_max_jerk,
        )
        before_target = self.target.position.copy()
        before_interceptor = self.interceptor.position.copy()
        before_net = None if self.net_position is None else self.net_position.copy()
        self.target.step(target_acceleration, self.config.physics_dt)
        self.interceptor.step(interceptor_acceleration, self.config.physics_dt)
        self.last_target_acceleration = target_acceleration
        self.last_interceptor_acceleration = interceptor_acceleration
        self.time += self.config.physics_dt

        if self.net_position is not None and self.net_velocity is not None:
            self.net_velocity += np.array([0.0, 0.0, -GRAVITY]) * self.config.physics_dt
            self.net_position += self.net_velocity * self.config.physics_dt
            relative_start = before_net - before_target
            relative_end = self.net_position - self.target.position
            self.min_net_distance = min(
                self.min_net_distance,
                segment_distance(relative_start, relative_end),
            )
            if is_hit(relative_start, relative_end, self.config.net_capture_radius):
                self.capture_time = self.time
                return "hit"
            if (
                np.linalg.norm(self.net_position) > self.config.sphere_radius
            ):
                return "miss"

        if np.linalg.norm(self.target.position) > self.config.sphere_radius:
            return "target_exit"
        if np.linalg.norm(self.interceptor.position) > self.config.sphere_radius:
            return "interceptor_exit"
        return None

    def target_acceleration(self) -> np.ndarray:
        # a_T_ref = k_v * (v_T_route - v_T)
        #         + A_horizontal * sin(omega_horizontal * t + phi_horizontal) * e_horizontal
        #         + A_vertical * sin(omega_vertical * t + phi_vertical) * e_vertical
        #         + eta_OU
        # phi_horizontal = phi_vertical = 0.0 and are not randomized.
        correlation = self.config.target_ou_correlation_time
        noise_scale = self.config.target_ou_std * np.sqrt(2.0 * self.config.physics_dt / correlation)
        self.ou_noise += (-self.ou_noise / correlation) * self.config.physics_dt + noise_scale * self.rng.normal(size=3)
        reference = self.config.target_speed * self.route_direction
        manoeuvre = (
            self.horizontal_amplitude * np.sin(self.horizontal_frequency * self.time) * self.route_horizontal_axis
            + self.vertical_amplitude * np.sin(self.vertical_frequency * self.time) * self.route_vertical_axis
        )
        command = self.config.target_speed_gain * (reference - self.target.velocity) + manoeuvre + self.ou_noise
        return clip_norm(command, self.config.target_max_acceleration)

    def jerk_limited(self, desired: np.ndarray, previous: np.ndarray, max_acceleration: float, max_jerk: float) -> np.ndarray:
        delta = clip_norm(np.asarray(desired) - previous, max_jerk * self.config.physics_dt)
        return clip_norm(previous + delta, max_acceleration)

    def observation(self) -> np.ndarray:
        values = np.concatenate(
            [
                (self.target.position - self.interceptor.position) / self.config.sphere_radius,
                (self.target.velocity - self.interceptor.velocity) / self.config.target_max_speed,
                self.interceptor.position / self.config.sphere_radius,
                self.interceptor.velocity / self.config.interceptor_max_speed,
                self.interceptor.quaternion,
                self.interceptor.angular_velocity / 10.0,
                self.last_interceptor_acceleration / self.config.interceptor_max_acceleration,
                np.array([self.time / self.config.reference_time]),
                np.array([float(self.launch_used)]),
                np.array([float(self.launch_gate_open())]),
                np.array([float(self.config.launch_curriculum_stage)]),
            ]
        )
        return values.astype(np.float32)

    def info(self) -> Dict[str, object]:
        return {
            "success": self.reason == "hit",
            "termination_reason": self.reason,
            "episode_time": self.time,
            "capture_time": self.capture_time,
            "min_distance": self.min_distance,
            "min_net_distance": self.min_net_distance,
            "control_effort": self.control_effort,
            "launch_distance": self.launch_distance,
            "launch_time": self.launch_time,
            "launch_used": self.launch_used,
            "launch_source": self.launch_source,
            "teacher_aim_action": self.launch_teacher_aim.copy(),
            "rule_aim": self.launch_source == "rule_aim",
            "rl_aim": self.launch_source == "rl_aim",
            "gate_ever_open": self.gate_ever_open,
            "gate_open_steps": self.gate_open_steps,
            "first_gate_time": self.first_gate_time,
            "curriculum_stage": self.config.launch_curriculum_stage,
            "action_mask": self.action_mask(),
        }

    def append_history(self) -> None:
        self.history["time"].append(self.time)
        self.history["target"].append(self.target.position.copy())
        self.history["interceptor"].append(self.interceptor.position.copy())
        net = np.full(3, np.nan) if self.net_position is None else self.net_position.copy()
        self.history["net"].append(net)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        before_time = self.time
        before_distance = float(np.linalg.norm(self.target.position - self.interceptor.position))
        gate_open = self.launch_gate_open()
        if gate_open:
            self.gate_open_steps += 1
            if not self.gate_ever_open:
                self.gate_ever_open = True
                self.first_gate_time = self.time
        desired_acceleration, launch_direction, launch_source = self.decode_action(action)
        launched_now = not self.launch_used and gate_open
        if launched_now:
            self.launch_teacher_aim = self.ballistic_aim_action()
            self.launch(launch_direction, launch_source)

        if launched_now:
            reason = self.resolve_launched_net()
        else:
            reason = None
            for substep in range(self.config.physics_substeps):
                reason = self.physics_step(desired_acceleration)
                if reason is not None:
                    break

        self.step_count += 1
        if not launched_now:
            self.append_history()
        after_distance = float(np.linalg.norm(self.target.position - self.interceptor.position))
        self.min_distance = min(self.min_distance, after_distance)
        self.reason = reason
        if not launched_now:
            self.control_effort += float(np.dot(self.last_interceptor_acceleration, self.last_interceptor_acceleration)) * self.config.control_dt
        elapsed_time = self.time - before_time
        time_penalty = self.config.time_penalty_scale * elapsed_time / self.config.reference_time
        approach_reward = 0.0
        if not launched_now:
            approach_reward = self.config.approach_progress_scale * (
                before_distance - after_distance
            ) / self.config.sphere_radius
        reward = approach_reward - time_penalty

        terminated = reason is not None
        truncated = False
        if reason == "hit":
            remaining_time_fraction = float(
                np.clip(1.0 - self.capture_time / self.config.reference_time, 0.0, 1.0)
            )
            reward = self.config.terminal_reward + self.config.capture_time_bonus * remaining_time_fraction
        elif reason == "miss":
            reward = (
                -self.config.miss_penalty
                + self.near_miss_reward()
                - time_penalty
                - self.remaining_time_penalty()
            )
        elif reason == "target_exit":
            if self.launch_used:
                reward = (
                    -self.config.miss_penalty
                    + self.near_miss_reward()
                    - time_penalty
                    - self.remaining_time_penalty()
                )
            else:
                reward = (
                    -self.config.passive_exit_penalty
                    - time_penalty
                    - self.remaining_time_penalty()
                )
        elif reason == "interceptor_exit":
            reward = (
                -self.config.interceptor_exit_penalty
                - time_penalty
                - self.remaining_time_penalty()
            )
        elif self.config.debug_max_steps and self.step_count >= self.config.debug_max_steps:
            # Never use this in the thesis result.  It is a true truncation,
            # so PPO may bootstrap from the returned final observation.
            truncated = True
            self.reason = "debug_guard"
        return self.observation(), float(reward), terminated, truncated, self.info()

    def remaining_time_penalty(self) -> float:
        remaining_fraction = float(
            np.clip(1.0 - self.time / self.config.reference_time, 0.0, 1.0)
        )
        return self.config.time_penalty_scale * remaining_fraction

    def near_miss_reward(self) -> float:
        if not np.isfinite(self.min_net_distance):
            return 0.0
        closeness = 1.0 - self.min_net_distance / self.config.near_miss_distance
        return self.config.near_miss_bonus * float(np.clip(closeness, 0.0, 1.0))

    def trajectory(self) -> Dict[str, np.ndarray]:
        return {name: np.asarray(values) for name, values in self.history.items()}

    def heuristic_action(self) -> np.ndarray:
        """Deterministic feasibility check with the analytic ballistic direction."""

        aim = self.ballistic_aim_action()
        if self.config.mode == "pn":
            return aim
        return np.concatenate([np.zeros(3, dtype=np.float32), aim])
