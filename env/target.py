"""Fixed, non-learning target maneuver generator.

The target only uses its own position and velocity.  In particular, it never
receives interceptor state, future noise, or an RL action.  Episode seeds only
change its sine phases and its temporally correlated OU-noise realization.
"""

import numpy as np


def _as_vector3(value, name):
    """Return a three-dimensional floating-point vector."""
    return np.asarray(value, dtype=np.float64)


def _clip_vector_norm(vector, maximum_norm):
    """Clip a vector magnitude while retaining its direction."""
    if maximum_norm is None:
        return vector

    norm = float(np.linalg.norm(vector))
    if norm > maximum_norm and norm > 0.0:
        return vector * (maximum_norm / norm)
    return vector


class FixedTargetManeuver:
    """Generate a reproducible, bounded 3-D target acceleration command.

    The raw command is

    ``a_goal + A_l sin(w_l t + phi_l) e_l + A_v sin(w_v t + phi_v) e_v + eta``,

    where ``eta`` is stationary Ornstein-Uhlenbeck acceleration noise.  It is
    then passed through acceleration, jerk, predicted-speed, and climb-rate
    limits before being returned.

    Frequencies are angular frequencies in rad/s.  Amplitudes, acceleration,
    and jerk use SI-style units (m/s^2 and m/s^3 when the environment uses
    metres and seconds).
    """

    def __init__(
        self,
        asset_position=(0.0, 0.0, 0.0),
        goal_acceleration=1.5,
        lateral_amplitude=1.0,
        vertical_amplitude=0.75,
        lateral_frequency=0.6,
        vertical_frequency=0.9,
        noise_correlation_time=1.0,
        noise_std=0.2,
        max_speed=15.0,
        max_acceleration=6.0,
        max_jerk=12.0,
        max_climb_rate=4.0,
        goal_speed=None,
        goal_speed_gain=0.0,
        seed=None,
        lateral_omega=None,
        vertical_omega=None,
    ):
        """Create one target generator with fixed maneuver parameters.

        ``lateral_omega`` and ``vertical_omega`` are accepted as explicit
        aliases for the two frequency arguments because the research notation
        writes them as ``omega_l`` and ``omega_v``.

        ``goal_speed`` is optional.  When supplied with a positive
        ``goal_speed_gain``, it adds a velocity-tracking term toward the asset;
        leaving it as ``None`` gives the simpler constant goal acceleration
        specified in the handover document.
        """
        if lateral_omega is not None:
            lateral_frequency = lateral_omega
        if vertical_omega is not None:
            vertical_frequency = vertical_omega

        self.asset_position = _as_vector3(asset_position, "asset_position")
        self.goal_acceleration = self._non_negative(goal_acceleration, "goal_acceleration")
        self.lateral_amplitude = self._non_negative(lateral_amplitude, "lateral_amplitude")
        self.vertical_amplitude = self._non_negative(vertical_amplitude, "vertical_amplitude")
        self.lateral_frequency = self._non_negative(lateral_frequency, "lateral_frequency")
        self.vertical_frequency = self._non_negative(vertical_frequency, "vertical_frequency")
        self.noise_correlation_time = self._non_negative(
            noise_correlation_time, "noise_correlation_time"
        )
        self.noise_std = self._non_negative(noise_std, "noise_std")
        self.max_speed = self._positive_or_none(max_speed, "max_speed")
        self.max_acceleration = self._positive_or_none(max_acceleration, "max_acceleration")
        self.max_jerk = self._positive_or_none(max_jerk, "max_jerk")
        self.max_climb_rate = self._positive_or_none(max_climb_rate, "max_climb_rate")
        self.goal_speed = self._positive_or_none(goal_speed, "goal_speed")
        self.goal_speed_gain = self._non_negative(goal_speed_gain, "goal_speed_gain")

        self._rng = np.random.default_rng(seed)
        self.time = 0.0
        self.lateral_phase = 0.0
        self.vertical_phase = 0.0
        self.noise_acceleration = np.zeros(3, dtype=np.float64)
        self.previous_acceleration = np.zeros(3, dtype=np.float64)
        self.last_raw_acceleration = np.zeros(3, dtype=np.float64)
        self.last_acceleration = np.zeros(3, dtype=np.float64)
        self.last_goal_acceleration = np.zeros(3, dtype=np.float64)
        self.last_lateral_acceleration = np.zeros(3, dtype=np.float64)
        self.last_vertical_acceleration = np.zeros(3, dtype=np.float64)

        self.reset(seed=seed)

    @staticmethod
    def _non_negative(value, name):
        return float(value)

    @staticmethod
    def _positive_or_none(value, name):
        if value is None:
            return None
        return float(value)

    def reset(self, seed=None):
        """Start a new episode and sample only phase/noise-randomness.

        Passing the same seed reproduces the complete command sequence for
        identical target state inputs.  A reset without a seed keeps consuming
        this object's private RNG, matching Gymnasium's usual episode-seeding
        behavior without perturbing NumPy's global random state.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.time = 0.0
        self.lateral_phase = float(self._rng.uniform(0.0, 2.0 * np.pi))
        self.vertical_phase = float(self._rng.uniform(0.0, 2.0 * np.pi))

        # Starting from the stationary distribution avoids a special zero-noise
        # transient at the beginning of every episode.
        self.noise_acceleration = self._rng.normal(0.0, self.noise_std, size=3)
        self.previous_acceleration = np.zeros(3, dtype=np.float64)
        self.last_raw_acceleration = np.zeros(3, dtype=np.float64)
        self.last_acceleration = np.zeros(3, dtype=np.float64)
        self.last_goal_acceleration = np.zeros(3, dtype=np.float64)
        self.last_lateral_acceleration = np.zeros(3, dtype=np.float64)
        self.last_vertical_acceleration = np.zeros(3, dtype=np.float64)

    def step(self, position, velocity, dt):
        """Return the target's next acceleration command.

        Parameters are the target's current world-frame position and velocity
        and one positive control/physics timestep.  The method does not update
        position or velocity itself; RotorPy (or the enclosing environment)
        remains the sole dynamics integrator.
        """
        position = _as_vector3(position, "position")
        velocity = _as_vector3(velocity, "velocity")
        dt = float(dt)

        forward_axis, lateral_axis, vertical_axis = self._local_frame(position, velocity)
        self._update_ou_noise(dt)

        self.last_goal_acceleration = self._goal_acceleration(forward_axis, velocity)
        self.last_lateral_acceleration = (
            self.lateral_amplitude
            * np.sin(self.lateral_frequency * self.time + self.lateral_phase)
            * lateral_axis
        )
        self.last_vertical_acceleration = (
            self.vertical_amplitude
            * np.sin(self.vertical_frequency * self.time + self.vertical_phase)
            * vertical_axis
        )
        self.last_raw_acceleration = (
            self.last_goal_acceleration
            + self.last_lateral_acceleration
            + self.last_vertical_acceleration
            + self.noise_acceleration
        )

        acceleration = self._apply_motion_limits(self.last_raw_acceleration, velocity, dt)
        self.previous_acceleration = acceleration.copy()
        self.last_acceleration = acceleration.copy()
        self.time += dt
        return acceleration.astype(np.float64, copy=False)

    def _local_frame(self, position, velocity):
        """Build forward/lateral/vertical maneuver axes at the target state."""
        to_asset = self.asset_position - position
        to_asset_norm = float(np.linalg.norm(to_asset))
        if to_asset_norm > 1e-8:
            forward_axis = to_asset / to_asset_norm
        else:
            # At the singular point, oppose the current trajectory when
            # possible.  This keeps a finite command without leaking any
            # interceptor information into the target model.
            speed = float(np.linalg.norm(velocity))
            if speed > 1e-8:
                forward_axis = -velocity / speed
            else:
                forward_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        lateral_axis = np.cross(world_up, forward_axis)
        lateral_norm = float(np.linalg.norm(lateral_axis))
        if lateral_norm <= 1e-8:
            # A near-vertical target-to-asset direction has no unique
            # horizontal lateral axis.  Choose a fixed, reproducible one.
            lateral_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            lateral_axis /= lateral_norm

        vertical_axis = np.cross(forward_axis, lateral_axis)
        vertical_norm = float(np.linalg.norm(vertical_axis))
        if vertical_norm <= 1e-8:
            # This branch is defensive; the construction above should already
            # make the axes orthogonal.
            vertical_axis = world_up.copy()
        else:
            vertical_axis /= vertical_norm
        return forward_axis, lateral_axis, vertical_axis

    def _goal_acceleration(self, forward_axis, velocity):
        """Return acceleration that makes the target continue toward the asset."""
        acceleration = self.goal_acceleration * forward_axis
        if self.goal_speed is not None and self.goal_speed_gain > 0.0:
            desired_velocity = self.goal_speed * forward_axis
            acceleration = acceleration + self.goal_speed_gain * (desired_velocity - velocity)
        return acceleration

    def _update_ou_noise(self, dt):
        """Advance a stationary OU acceleration process by one timestep."""
        if self.noise_std == 0.0:
            self.noise_acceleration.fill(0.0)
            return

        if self.noise_correlation_time == 0.0:
            self.noise_acceleration = self._rng.normal(0.0, self.noise_std, size=3)
            return

        correlation = float(np.exp(-dt / self.noise_correlation_time))
        innovation_std = self.noise_std * np.sqrt(max(0.0, 1.0 - correlation**2))
        innovation = self._rng.normal(0.0, innovation_std, size=3)
        self.noise_acceleration = correlation * self.noise_acceleration + innovation

    def _apply_motion_limits(self, raw_acceleration, velocity, dt):
        """Apply the command limits in an order that preserves smooth motion.

        The velocity/climb projections are predictive: they constrain the
        Euler estimate ``velocity + acceleration * dt``.  RotorPy is still the
        physical integrator, so these are command safeguards rather than a
        replacement for its own velocity limits.
        """
        acceleration = raw_acceleration.copy()

        # Bring the desired next velocity inside the speed and climb envelopes
        # first.  Zero acceleration is safe whenever the current state itself
        # is within those envelopes, so the later norm projection remains safe.
        acceleration = self._limit_predicted_climb_rate(acceleration, velocity, dt)
        acceleration = self._limit_predicted_speed(acceleration, velocity, dt)
        acceleration = _clip_vector_norm(acceleration, self.max_acceleration)

        # Limit the step-to-step change after magnitude limiting.  Both this
        # command and the previous command are inside the acceleration ball,
        # so their interpolation stays within the acceleration limit.
        acceleration = self._limit_jerk(acceleration, dt)

        # At normal feasible states the preceding command already satisfies
        # the predictive limits.  Reproject once to handle a controller whose
        # measured velocity has drifted from the prior command prediction.
        projected_acceleration = self._limit_predicted_climb_rate(acceleration, velocity, dt)
        projected_acceleration = self._limit_predicted_speed(projected_acceleration, velocity, dt)

        # If the required speed correction is compatible with the physical
        # actuator limits, use it.  For an externally supplied state already
        # outside a limit, all constraints can be mutually infeasible; then
        # preserve the physical acceleration and jerk caps.
        projected_acceleration = _clip_vector_norm(projected_acceleration, self.max_acceleration)
        jerk_limited = self._limit_jerk(projected_acceleration, dt)
        if self._predicted_motion_is_within_limits(jerk_limited, velocity, dt):
            acceleration = jerk_limited

        return acceleration

    def _limit_jerk(self, acceleration, dt):
        if self.max_jerk is None:
            return acceleration

        delta = acceleration - self.previous_acceleration
        return self.previous_acceleration + _clip_vector_norm(delta, self.max_jerk * dt)

    def _limit_predicted_speed(self, acceleration, velocity, dt):
        if self.max_speed is None:
            return acceleration

        predicted_velocity = velocity + acceleration * dt
        predicted_speed = float(np.linalg.norm(predicted_velocity))
        if predicted_speed > self.max_speed:
            predicted_velocity *= self.max_speed / predicted_speed
            return (predicted_velocity - velocity) / dt
        return acceleration

    def _limit_predicted_climb_rate(self, acceleration, velocity, dt):
        if self.max_climb_rate is None:
            return acceleration

        acceleration = acceleration.copy()
        predicted_vertical_speed = velocity[2] + acceleration[2] * dt
        clipped_vertical_speed = float(
            np.clip(predicted_vertical_speed, -self.max_climb_rate, self.max_climb_rate)
        )
        acceleration[2] = (clipped_vertical_speed - velocity[2]) / dt
        return acceleration

    def _predicted_motion_is_within_limits(self, acceleration, velocity, dt):
        tolerance = 1e-9
        predicted_velocity = velocity + acceleration * dt
        if self.max_speed is not None and np.linalg.norm(predicted_velocity) > self.max_speed + tolerance:
            return False
        if self.max_climb_rate is not None and abs(predicted_velocity[2]) > self.max_climb_rate + tolerance:
            return False
        return True


# The aliases make the module convenient to import without duplicating any
# maneuver implementation.  They all deliberately share one fixed generator.
TargetManeuver = FixedTargetManeuver
TargetManeuverGenerator = FixedTargetManeuver
