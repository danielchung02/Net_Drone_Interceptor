"""Three-dimensional proportional-navigation guidance.

This module intentionally only converts relative kinematics into a guidance
acceleration.  It does not know about the Gymnasium environment, rewards, or
payload state, so the same PN implementation can be used for a baseline and
for environment smoke tests.
"""

import numpy as np


def _as_vector3(value, name):
    """Return *value* as a finite three-dimensional floating-point vector."""
    return np.asarray(value, dtype=np.float64)


def _clip_vector_norm(vector, maximum_norm):
    """Clip a vector magnitude while retaining its direction."""
    if maximum_norm is None:
        return vector

    norm = float(np.linalg.norm(vector))
    if norm > maximum_norm and norm > 0.0:
        return vector * (maximum_norm / norm)
    return vector


def proportional_navigation(
    relative_position,
    relative_velocity,
    navigation_constant=3.0,
    max_acceleration=None,
    minimum_range=1e-6,
    require_closing=True,
):
    """Compute a bounded 3-D proportional-navigation acceleration.

    Parameters
    ----------
    relative_position : array-like, shape (3,)
        ``target_position - interceptor_position`` in world coordinates.
    relative_velocity : array-like, shape (3,)
        ``target_velocity - interceptor_velocity`` in world coordinates.
    navigation_constant : float, default=3.0
        Dimensionless PN gain.  Values around 3--5 are common starting
        points; its final value should be chosen through feasibility tests.
    max_acceleration : float or None, default=None
        Magnitude cap for the returned command.  ``None`` means no cap.
    minimum_range : float, default=1e-6
        Near-zero LOS range below which the function returns zero rather than
        dividing by an ill-conditioned range squared.
    require_closing : bool, default=True
        If true, return zero once the target is not closing.  This avoids a
        large, physically meaningless command for a receding target.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Lateral guidance acceleration in world coordinates.

    Notes
    -----
    With ``r = p_T - p_I``, ``v = v_T - v_I`` and positive closing speed
    ``V_c = -r_hat dot v``, the implemented command is

    ``a_PN = N * V_c * (omega_LOS cross r_hat)``,

    where ``omega_LOS = (r cross v) / ||r||^2``.  The expression is fully
    three-dimensional and is perpendicular to the line of sight.
    """
    relative_position = _as_vector3(relative_position, "relative_position")
    relative_velocity = _as_vector3(relative_velocity, "relative_velocity")

    if not np.isfinite(relative_position).all() or not np.isfinite(relative_velocity).all():
        return np.zeros(3, dtype=np.float64)

    navigation_constant = float(navigation_constant)
    minimum_range = float(minimum_range)
    if max_acceleration is not None:
        max_acceleration = float(max_acceleration)

    range_to_target = float(np.linalg.norm(relative_position))
    if range_to_target <= minimum_range:
        return np.zeros(3, dtype=np.float64)

    line_of_sight = relative_position / range_to_target
    closing_speed = -float(np.dot(line_of_sight, relative_velocity))
    if require_closing and closing_speed <= 0.0:
        return np.zeros(3, dtype=np.float64)

    # Keeping the vector formula makes the sign convention explicit:
    # a positive LOS rotation turns the interceptor in the same direction.
    los_angular_rate = np.cross(relative_position, relative_velocity) / (range_to_target**2)
    acceleration = navigation_constant * closing_speed * np.cross(los_angular_rate, line_of_sight)
    acceleration = _clip_vector_norm(acceleration, max_acceleration)

    if not np.isfinite(acceleration).all():
        return np.zeros(3, dtype=np.float64)
    return acceleration.astype(np.float64, copy=False)


def compute_pn_acceleration(*args, **kwargs):
    """Readable alias for :func:`proportional_navigation`."""
    return proportional_navigation(*args, **kwargs)


class ProportionalNavigation:
    """State-free PN guidance object with fixed tuning parameters.

    The class form keeps the environment call site short while preserving the
    standalone :func:`proportional_navigation` function for direct tests.
    """

    def __init__(
        self,
        navigation_constant=3.0,
        max_acceleration=None,
        minimum_range=1e-6,
        require_closing=True,
    ):
        self.navigation_constant = float(navigation_constant)
        self.max_acceleration = max_acceleration
        self.minimum_range = float(minimum_range)
        self.require_closing = bool(require_closing)

    def compute(self, relative_position, relative_velocity):
        """Return the PN command for the supplied relative state."""
        return proportional_navigation(
            relative_position=relative_position,
            relative_velocity=relative_velocity,
            navigation_constant=self.navigation_constant,
            max_acceleration=self.max_acceleration,
            minimum_range=self.minimum_range,
            require_closing=self.require_closing,
        )

    def step(self, relative_position, relative_velocity):
        """Alias of :meth:`compute` for control-loop style call sites."""
        return self.compute(relative_position, relative_velocity)

    def __call__(self, relative_position, relative_velocity):
        return self.compute(relative_position, relative_velocity)


PNGuidance = ProportionalNavigation
