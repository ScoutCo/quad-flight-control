"""Unit-style tests for the cascaded controller building block."""

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
from sixdof_sim.math_utils import quat_to_rotation_matrix
from sixdof_sim.parameters import ActuatorParameters, ControllerGains, VehicleParameters
from sixdof_sim.states import SixDofState


def make_controller() -> CascadedController:
    return CascadedController(
        VehicleParameters(),
        ControllerGains(),
        ActuatorParameters(),
    )


def test_hover_regulation() -> None:
    controller = make_controller()
    state = SixDofState(position_ned=np.zeros(3), velocity_ned=np.zeros(3))
    command = PositionVelocityCommand(
        position_ned=np.zeros(3),
        velocity_ned_ff=np.zeros(3),
        accel_ned_ff=np.zeros(3),
        yaw_heading=0.0,
    )

    result = controller.update(state, command, gravity=np.array([0.0, 0.0, 9.80665]), dt=0.01)
    assert math.isclose(
        result.thrust_command,
        controller.vehicle.mass * 9.80665,
        rel_tol=1e-3,
    )


def test_constant_acceleration_feedforward() -> None:
    controller = make_controller()
    desired_acc = np.array([0.5, -0.3, 0.2])
    state = SixDofState(position_ned=np.zeros(3), velocity_ned=np.zeros(3))
    command = PositionVelocityCommand(
        position_ned=np.zeros(3),
        velocity_ned_ff=np.zeros(3),
        accel_ned_ff=desired_acc,
        yaw_heading=0.0,
    )

    result = controller.update(state, command, gravity=np.array([0.0, 0.0, 9.80665]), dt=0.01)
    thrust_vec = controller.vehicle.mass * (desired_acc - np.array([0.0, 0.0, 9.80665]))
    assert np.allclose(
        thrust_vec / np.linalg.norm(thrust_vec),
        -result.body_rate_command / np.linalg.norm(result.body_rate_command),
        atol=1e-3,
    )


def test_tilt_limit_respected() -> None:
    controller = make_controller()
    gains = controller.gains
    assert gains.max_tilt_angle_rad > 0.0

    large_acc = np.array([20.0, 0.0, 0.0])
    state = SixDofState(position_ned=np.zeros(3), velocity_ned=np.zeros(3))
    command = PositionVelocityCommand(
        position_ned=np.zeros(3),
        velocity_ned_ff=np.zeros(3),
        accel_ned_ff=large_acc,
        yaw_heading=0.0,
    )

    result = controller.update(state, command, gravity=np.array([0.0, 0.0, 9.80665]), dt=0.01)
    rotation = quat_to_rotation_matrix(result.attitude_target_quat)
    z_body = rotation[:, 2]
    tilt = math.acos(min(max(-z_body[2], -1.0), 1.0))
    assert tilt <= gains.max_tilt_angle_rad + 1e-3
