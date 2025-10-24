from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class State:
    """State tracked by the simulator."""

    position_ned: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    velocity_ned: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    quaternion_bn: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    )
    angular_velocity_body: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))

    def copy(self) -> "State":
        return State(
            position_ned=self.position_ned.copy(),
            velocity_ned=self.velocity_ned.copy(),
            quaternion_bn=self.quaternion_bn.copy(),
            angular_velocity_body=self.angular_velocity_body.copy(),
        )
