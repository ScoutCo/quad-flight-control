"""Closed-loop tests that exercise the cascaded controller with a simple plant."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from sixdof_sim.commands import PositionVelocityCommand
from sixdof_sim.control import CascadedController
from sixdof_sim.parameters import ActuatorParameters, ControllerGains, VehicleParameters
from sixdof_sim.states import SixDofState


def make_controller() -> CascadedController:
    return CascadedController(
        VehicleParameters(),
        ControllerGains(),
        ActuatorParameters(),
    )


def simple_plant_step(
    state: SixDofState,
    accel_cmd: np.ndarray,
    dt: float,
    max_thrust: float,
) -> SixDofState:
    s = state.copy()
    accel = np.clip(accel_cmd, -max_thrust, max_thrust)
    s.velocity_ned += accel * dt
    s.position_ned += s.velocity_ned * dt
    return s


def run_closed_loop(
    controller: CascadedController,
    command: PositionVelocityCommand,
    steps: int,
    dt: float,
    disturbance: np.ndarray | None = None,
):
    state = SixDofState(position_ned=np.zeros(3), velocity_ned=np.zeros(3))
    for _ in range(steps):
        ctrl = controller.update(
            state, command, gravity=np.array([0.0, 0.0, 9.80665]), dt=dt
        )
        accel_cmd = ctrl.velocity_target.copy()
        if disturbance is not None:
            accel_cmd += disturbance
        state = simple_plant_step(state, accel_cmd, dt, max_thrust=10.0)
    return state, ctrl


def test_step_response_saturation() -> None:
    controller = make_controller()
    command = PositionVelocityCommand(
        position_ned=np.array([5.0, 0.0, 0.0]),
        velocity_ned_ff=np.zeros(3),
        accel_ned_ff=np.zeros(3),
        yaw_heading=0.0,
    )
    state, ctrl = run_closed_loop(controller, command, steps=200, dt=0.01)
    assert np.linalg.norm(ctrl.velocity_target[:2]) <= controller.gains.max_velocity_target_xy + 1e-3


def test_constant_disturbance_anti_windup() -> None:
    controller = make_controller()
    command = PositionVelocityCommand(
        position_ned=np.zeros(3),
        velocity_ned_ff=np.zeros(3),
        accel_ned_ff=np.zeros(3),
        yaw_heading=0.0,
    )
    disturbance = np.array([0.0, 0.0, 5.0])  # constant downward accel
    state, ctrl = run_closed_loop(controller, command, steps=300, dt=0.01, disturbance=disturbance)
    assert abs(ctrl.thrust_command) >= controller.actuator_params.max_thrust * 0.9


def test_curvature_feedforward_tracking() -> None:
    controller = make_controller()
    omega = 0.5
    radius = 30.0
    accel_ff = np.array([-(omega**2) * radius, 0.0, 0.0])
    command = PositionVelocityCommand(
        position_ned=np.zeros(3),
        velocity_ned_ff=np.array([0.0, omega * radius, 0.0]),
        accel_ned_ff=accel_ff,
        yaw_heading=0.0,
    )
    state, ctrl = run_closed_loop(controller, command, steps=500, dt=0.01)
    assert np.linalg.norm(ctrl.velocity_target[:2]) <= controller.gains.max_accel_target_xy + 1e-3
