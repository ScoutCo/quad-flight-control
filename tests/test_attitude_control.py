"""Tests focusing on the attitude/rate portion of the controller."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from sixdof_sim import PositionVelocityCommand, SixDofSimulator
from sixdof_sim.math_utils import quat_to_euler


def test_roll_command_converges() -> None:
    sim = SixDofSimulator(dt=0.01)
    desired_roll = math.radians(20.0)

    def command_fn(_t: float, _state):
        cmd = PositionVelocityCommand(
            position_ned=np.zeros(3),
            velocity_ned_ff=np.zeros(3),
            accel_ned_ff=np.array([9.80665 * math.tan(desired_roll), 0.0, 0.0]),
            yaw_heading=0.0,
        )
        return cmd

    history = sim.run(5.0, command_fn)
    roll, pitch, _ = quat_to_euler(history[-1].state.quaternion_bn)
    assert abs(roll - desired_roll) < math.radians(2.0)


def test_negative_roll_command_converges() -> None:
    sim = SixDofSimulator(dt=0.01)
    desired_roll = math.radians(-15.0)

    def command_fn(_t: float, _state):
        cmd = PositionVelocityCommand(
            position_ned=np.zeros(3),
            velocity_ned_ff=np.zeros(3),
            accel_ned_ff=np.array([9.80665 * math.tan(desired_roll), 0.0, 0.0]),
            yaw_heading=0.0,
        )
        return cmd

    history = sim.run(5.0, command_fn)
    roll, pitch, _ = quat_to_euler(history[-1].state.quaternion_bn)
    assert abs(roll - desired_roll) < math.radians(2.0)
