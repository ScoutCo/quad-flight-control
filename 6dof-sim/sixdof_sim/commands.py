from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class PositionVelocityCommand:
    position_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity_ned_ff: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw_heading: float | None = None

    def copy(self) -> PositionVelocityCommand:
        return PositionVelocityCommand(
            position_ned=self.position_ned.copy(),
            velocity_ned_ff=self.velocity_ned_ff.copy(),
            yaw_heading=self.yaw_heading,
        )
