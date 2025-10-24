from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable
import numpy as np

from .actuators import ActuatorDynamics, ActuatorState
from .commands import PositionVelocityCommand
from .control import CascadedController, ControllerState
from .dynamics import RigidBodyDynamics
from .environment import Environment
from .parameters import (
    ActuatorParameters,
    ControllerGains,
    EnvironmentParameters,
    VehicleParameters,
)
from .states import SixDofState


@dataclass
class SimulationStepResult:
    time: float
    state: SixDofState
    controller: ControllerState
    actuators: ActuatorState


class SixDofSimulator:
    def __init__(
        self,
        vehicle: VehicleParameters | None = None,
        controller: ControllerGains | None = None,
        actuators: ActuatorParameters | None = None,
        environment: EnvironmentParameters | None = None,
        dt: float = 0.01,
    ) -> None:
        self.vehicle = vehicle or VehicleParameters()
        self.controller_gains = controller or ControllerGains()
        self.actuator_params = actuators or ActuatorParameters()
        self.environment_params = environment or EnvironmentParameters()
        self.dt = dt

        self.dynamics = RigidBodyDynamics(self.vehicle)
        self.controller = CascadedController(
            self.vehicle, self.controller_gains, self.actuator_params
        )
        self.actuators = ActuatorDynamics(self.actuator_params)
        self.environment = Environment(self.environment_params)

        self.state = SixDofState()
        self.time = 0.0

    def reset(self, state: SixDofState | None = None, time: float = 0.0) -> None:
        self.state = state.copy() if state is not None else SixDofState()
        self.time = time
        self.controller.reset()
        self.actuators.reset()

    def step(self, command: PositionVelocityCommand, dt: float | None = None) -> SimulationStepResult:
        dt = dt or self.dt
        env_gravity = self.environment.gravity_vector()
        wind = self.environment.wind_ned(self.time)
        force_dist = self.environment.external_force_body(self.time)
        moment_dist = self.environment.external_moment_body(self.time)

        ctrl_state = self.controller.update(self.state, command, env_gravity, dt)
        actuator_state = self.actuators.step(ctrl_state.thrust_command, ctrl_state.moment_command, dt)

        next_state = self.dynamics.rk4_step(
            self.state,
            thrust=actuator_state.thrust,
            moment_body=actuator_state.moment_body,
            env_gravity=env_gravity,
            wind_ned=wind,
            external_force_body=force_dist,
            external_moment_body=moment_dist,
            dt=dt,
        )

        self.state = next_state
        self.time += dt
        return SimulationStepResult(time=self.time, state=next_state.copy(), controller=ctrl_state, actuators=actuator_state)

    def run(
        self,
        final_time: float,
        command_fn: Callable[[float, SixDofState], PositionVelocityCommand],
        progress_callback: Callable[[SimulationStepResult], None] | None = None,
    ) -> list[SimulationStepResult]:
        steps = int(np.ceil((final_time - self.time) / self.dt))
        history: list[SimulationStepResult] = []
        for _ in range(steps):
            cmd = command_fn(self.time, self.state.copy())
            result = self.step(cmd, dt=self.dt)
            history.append(result)
            if progress_callback is not None:
                progress_callback(result)
        return history
