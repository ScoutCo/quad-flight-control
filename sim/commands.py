from dataclasses import dataclass, field
from typing import Self

import numpy as np


@dataclass
class PositionVelocityCommand:
    """Position/velocity/acceleration command passed to the controller."""

    position_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity_ned_ff: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accel_ned_ff: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw_heading: float | None = None

    def copy(self) -> Self:
        return PositionVelocityCommand(
            position_ned=self.position_ned.copy(),
            velocity_ned_ff=self.velocity_ned_ff.copy(),
            accel_ned_ff=self.accel_ned_ff.copy(),
            yaw_heading=self.yaw_heading,
        )
