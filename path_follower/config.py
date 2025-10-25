from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PathFollowerConfig:
    """Configuration values for the position/velocity path follower.

    Times are expressed in seconds; distances and velocities are in SI units.
    """

    lookahead_offset_s: float = 0.5
    max_plan_age_s: float = 1.0
    max_plan_future_s: float = 5.0
    tau_v_xy: float = 0.2
    tau_v_z: float = 0.2
    a_max_xy: float = 3.5
    a_max_up: float = 2.5
    a_max_down: float = 2.0
    v_max_xy: float = 15.0
    v_max_up: float = 4.0
    v_max_down: float = 4.0

    def __post_init__(self) -> None:
        if self.lookahead_offset_s < 0.0:
            raise ValueError("lookahead_offset_s must be non-negative.")
        if self.max_plan_age_s <= 0.0:
            raise ValueError("max_plan_age_s must be positive.")
        if self.max_plan_future_s <= 0.0:
            raise ValueError("max_plan_future_s must be positive.")
        if self.tau_v_xy < 0.0 or self.tau_v_z < 0.0:
            raise ValueError("Velocity smoothing time constants must be non-negative.")
        if self.a_max_xy <= 0.0 or self.a_max_up <= 0.0 or self.a_max_down <= 0.0:
            raise ValueError("Acceleration limits must be strictly positive.")
        if self.v_max_xy <= 0.0 or self.v_max_up <= 0.0 or self.v_max_down <= 0.0:
            raise ValueError("Velocity limits must be strictly positive.")
