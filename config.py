"""Shared physical experiment settings, written with an explicit __init__."""

from pathlib import Path
from typing import Dict, Tuple


class ExperimentConfig:
    def __init__(self):
        # Experiment identity and common evaluation protocol.
        self.mode = "pn"
        self.seed = 0
        self.device = "auto"
        # None means that training continues until the process is interrupted.
        self.total_train_steps = None
        self.eval_interval_steps = 100_000
        self.save_interval_steps = 100_000
        self.n_eval_episodes = 20
        self.eval_seed_bank = tuple(range(10_000, 10_020))
        self.overwrite = False
        self.resume = False
        self.additional_train_steps = 0

        # RotorPy runs headless during training. ``simple`` is debugging only.
        self.physics_engine = "rotorpy"
        self.rotorpy_vehicle = "crazyflie"
        self.rotorpy_aero = True
        self.control_dt = 0.05 #RL 정책이 action을 새로 내는 주기
        self.physics_dt = self.control_dt / 3.0 #한 action마다 RotorPy physics substep 3회
        self.debug_max_steps = 0

        # Empty spherical engagement volume.
        self.sphere_radius = 100.0
        self.target_route_central_angle_degrees = 135.0

        # Fixed non-learning target controller.
        self.target_speed = 8.0
        # a_T_ref = k_v * (v_T_route - v_T)
        #         + A_horizontal * sin(omega_horizontal * t + phi_horizontal) * e_horizontal
        #         + A_vertical * sin(omega_vertical * t + phi_vertical) * e_vertical
        #         + eta_OU
        # phi_horizontal = phi_vertical = 0.0 and are not randomized.
        self.target_max_speed = 15.0
        self.target_max_acceleration = 8.0
        self.target_max_jerk = 30.0
        self.target_speed_limit_gain = 2.0
        self.target_speed_gain = 1.6 
        self.target_sine_horizontal_amplitude_range = (1.50, 2.50)
        self.target_sine_vertical_amplitude_range = (1.05, 1.75)
        self.target_sine_horizontal_frequency_range = (0.56, 0.84)
        self.target_sine_vertical_frequency_range = (0.36, 0.54)
        self.target_ou_std = 0.55 #Ornstein–Uhlenbeck의 약자
        self.target_ou_correlation_time = 0.8

        # Interceptor high-level command limits.
        self.interceptor_max_acceleration = 8.0
        self.interceptor_max_jerk = 30.0
        self.interceptor_max_speed = 15.0
        self.interceptor_speed_limit_gain = 2.0
        self.pn_navigation_constant = 4.0
        self.pn_speed_gain = 1.6

        # One abstract capture device.
        self.net_speed = 45.0 
        self.net_capture_radius = 2.0

        # Aim curriculum. PN starts at stage 1; E2E starts at stage 0.
        # 0: rule-based ballistic aim while E2E learns guidance.
        # 1: RL directly chooses LOS-relative launch angles.
        self.launch_curriculum_stage = 0
        self.curriculum_success_threshold = 0.80
        self.curriculum_required_evaluations = 2
        self.curriculum_success_streak = 0
        self.fixed_auto_launch_distance = 15.0
        self.launch_gate_min_closing_speed = 5.0
        self.launch_azimuth_limit = 0.5235987755982988  # 30 degrees
        self.launch_elevation_limit = 0.3490658503988659  # 20 degrees

        # Identical capture-focused reward for PN and end-to-end agents.
        self.terminal_reward = 10.0
        self.capture_time_bonus = 5.0
        self.time_penalty_scale = 2.5
        # Signed distance progress: moving away gives the opposite reward, so
        # hovering/following at a fixed distance cannot accumulate reward.
        self.approach_progress_scale = 5.0
        self.miss_penalty = 12.0
        self.passive_exit_penalty = 15.0
        self.interceptor_exit_penalty = 25.0
        self.near_miss_bonus = 4.0
        self.near_miss_distance = 40.0
        self.reference_time = 25.0 #2R/vt = 2*100/8 = 25
        self.run_root = "runs" 

    def validate(self) -> None:
        if self.mode not in {"pn", "e2e"}:
            raise ValueError("mode must be 'pn' or 'e2e'")
        if self.physics_engine not in {"rotorpy", "simple"}:
            raise ValueError("physics_engine must be 'rotorpy' or 'simple'")
        if self.control_dt <= 0.0 or self.physics_dt <= 0.0:
            raise ValueError("time steps must be positive")
        if abs(self.control_dt / self.physics_dt - round(self.control_dt / self.physics_dt)) > 1e-8:
            raise ValueError("physics_dt must divide control_dt exactly")
        if self.n_eval_episodes > len(self.eval_seed_bank):
            raise ValueError("evaluation seed bank is shorter than n_eval_episodes")
        if self.total_train_steps is not None and self.total_train_steps <= 0:
            raise ValueError("total_train_steps must be positive or None")
        if self.launch_curriculum_stage not in {0, 1}:
            raise ValueError("launch_curriculum_stage must be 0 or 1")
        for name in (
            "target_sine_horizontal_amplitude_range",
            "target_sine_vertical_amplitude_range",
            "target_sine_horizontal_frequency_range",
            "target_sine_vertical_frequency_range",
        ):
            lower, upper = getattr(self, name)
            if lower <= 0.0 or lower >= upper:
                raise ValueError("{} must satisfy 0 < minimum < maximum".format(name))

    @property
    def launch_curriculum_name(self) -> str:
        return ("guidance_with_rule_aim", "rl_los_aim")[self.launch_curriculum_stage]

    def update_launch_curriculum(self, success_rate: float) -> bool:
        """Advance only after repeated held-out success, never from a step count."""

        if self.launch_curriculum_stage >= 1:
            return False
        qualifies = success_rate >= self.curriculum_success_threshold
        if qualifies:
            self.curriculum_success_streak += 1
        else:
            self.curriculum_success_streak = 0
        if self.curriculum_success_streak < self.curriculum_required_evaluations:
            return False
        self.launch_curriculum_stage += 1
        self.curriculum_success_streak = 0
        return True

    def load_dict(self, values: Dict[str, object]) -> None:
        """Restore the fields understood by the current configuration."""

        for name, value in values.items():
            if not hasattr(self, name):
                continue
            if name == "eval_seed_bank":
                value = tuple(value)
            setattr(self, name, value)
        self.validate()

    @property
    def action_dim(self) -> int:
        if self.mode == "pn":
            # [LOS-relative horizontal launch angle, vertical launch angle]
            return 2
        # [LOS acceleration, horizontal acceleration, vertical acceleration,
        #  LOS-relative horizontal launch angle, vertical launch angle]
        return 5

    @property
    def physics_substeps(self) -> int:
        return int(round(self.control_dt / self.physics_dt))

    def agent_run_dir(self, agent_name: str) -> Path:
        return Path(self.run_root) / self.mode / agent_name

    def to_dict(self) -> Dict[str, object]:
        values = self.__dict__.copy()
        values["eval_seed_bank"] = list(self.eval_seed_bank)
        return values
