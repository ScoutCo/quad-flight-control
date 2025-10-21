from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .parameters import EnvironmentParameters


@dataclass
class Environment:
    params: EnvironmentParameters

    def gravity_vector(self) -> np.ndarray:
        return np.array([0.0, 0.0, self.params.gravity])

    def wind_ned(self, _time: float) -> np.ndarray:
        return self.params.wind_ned

    def external_force_body(self, _time: float) -> np.ndarray:
        return self.params.disturbance_force_body

    def external_moment_body(self, _time: float) -> np.ndarray:
        return self.params.disturbance_moment_body
