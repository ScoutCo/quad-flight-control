"""Tests that isolate the attitude / rate loop + actuator dynamics."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from sixdof_sim import PositionVelocityCommand, SixDofSimulator
from sixdof_sim.math_utils import quaternion_error


def max_orientation_error(history) -> float:
    max_angle = 0.0
    for step in history:
        diff = quaternion_error(step.controller.attitude_target_quat, step.state.quaternion_bn)
        angle = 2.0 * math.acos(max(min(diff[0], 1.0), -1.0))
        max_angle = max(max_angle, angle)
    return max_angle


def test_constant_x_accel() -> None:
    accel = np.array([2.0, 0.0, 0.0])
    sim = SixDofSimulator(dt=0.01)

    def command_fn(_t: float, _state):
        return PositionVelocityCommand(
            position_ned=np.zeros(3),
            velocity_ned_ff=np.zeros(3),
            accel_ned_ff=accel,
            yaw_heading=0.0,
        )

    history = sim.run(5.0, command_fn)
    err = max_orientation_error(history)
    assert err < math.radians(5.0)


def test_constant_y_accel() -> None:
    accel = np.array([0.0, 2.0, 0.0])
    sim = SixDofSimulator(dt=0.01)

    def command_fn(_t: float, _state):
        return PositionVelocityCommand(
            position_ned=np.zeros(3),
            velocity_ned_ff=np.zeros(3),
            accel_ned_ff=accel,
            yaw_heading=0.0,
        )

    history = sim.run(5.0, command_fn)
    err = max_orientation_error(history)
    assert err < math.radians(5.0)


def test_yaw_command() -> None:
    sim = SixDofSimulator(dt=0.01)
    desired_yaw = math.radians(90.0)

    def command_fn(_t: float, _state):
        return PositionVelocityCommand(
            position_ned=np.zeros(3),
            velocity_ned_ff=np.zeros(3),
            accel_ned_ff=np.zeros(3),
            yaw_heading=desired_yaw,
        )

    history = sim.run(5.0, command_fn)
    err = max_orientation_error(history)
    assert err < math.radians(5.0)
