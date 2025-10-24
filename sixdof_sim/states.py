from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SixDofState:
    position_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion_bn: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    angular_velocity_body: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def copy(self) -> "SixDofState":
        return SixDofState(
            position_ned=self.position_ned.copy(),
            velocity_ned=self.velocity_ned.copy(),
            quaternion_bn=self.quaternion_bn.copy(),
            angular_velocity_body=self.angular_velocity_body.copy(),
        )

    def as_vector(self) -> np.ndarray:
        return np.hstack(
            [
                self.position_ned,
                self.velocity_ned,
                self.quaternion_bn,
                self.angular_velocity_body,
            ]
        )

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "SixDofState":
        return cls(
            position_ned=vec[0:3].copy(),
            velocity_ned=vec[3:6].copy(),
            quaternion_bn=vec[6:10].copy(),
            angular_velocity_body=vec[10:13].copy(),
        )
