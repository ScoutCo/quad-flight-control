from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SimpleState:
    """State tracked by the simplified simulator."""

    position_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion_bn: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    )
    angular_velocity_body: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def copy(self) -> "SimpleState":
        return SimpleState(
            position_ned=self.position_ned.copy(),
            velocity_ned=self.velocity_ned.copy(),
            quaternion_bn=self.quaternion_bn.copy(),
            angular_velocity_body=self.angular_velocity_body.copy(),
        )
