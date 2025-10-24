from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .parameters import ActuatorParameters
from .math_utils import clamp


@dataclass
class ActuatorState:
    thrust: float = 0.0
    moment_body: np.ndarray = field(default_factory=lambda: np.zeros(3))


class ActuatorDynamics:
    def __init__(self, params: ActuatorParameters):
        self.params = params
        self.state = ActuatorState()

    def reset(self) -> None:
        self.state = ActuatorState()

    def step(
        self, thrust_cmd: float, moment_cmd: np.ndarray, dt: float
    ) -> ActuatorState:
        thrust_target = np.clip(
            thrust_cmd, self.params.thrust_min, self.params.max_thrust
        )
        moment_target = clamp(
            moment_cmd, -self.params.max_moment, self.params.max_moment
        )

        tau_t = self.params.thrust_time_constant
        tau_m = self.params.moment_time_constant

        thrust_dot = (thrust_target - self.state.thrust) / max(tau_t, 1e-6)
        moment_dot = (moment_target - self.state.moment_body) / max(tau_m, 1e-6)

        self.state.thrust += thrust_dot * dt
        self.state.moment_body += moment_dot * dt

        self.state.thrust = float(
            np.clip(self.state.thrust, self.params.thrust_min, self.params.max_thrust)
        )
        self.state.moment_body = clamp(
            self.state.moment_body, -self.params.max_moment, self.params.max_moment
        )
        return ActuatorState(
            thrust=self.state.thrust, moment_body=self.state.moment_body.copy()
        )
