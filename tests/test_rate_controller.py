"""Tests for the rate/attitude-to-moment loop and actuator saturation behaviour."""

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
from sixdof_sim.parameters import ActuatorParameters, ControllerGains


def run_sim(step_fn, duration: float = 3.0, dt: float = 0.002):
    sim = SixDofSimulator(dt=dt)
    return sim.run(duration, step_fn)


def test_rate_loop_tracks_roll_without_saturation() -> None:
    desired_roll = math.radians(15.0)

    def command_fn(_t: float, _state):
        return PositionVelocityCommand(
            position_ned=np.zeros(3),
            velocity_ned_ff=np.zeros(3),
            accel_ned_ff=np.array([9.80665 * math.tan(desired_roll), 0.0, 0.0]),
            yaw_heading=0.0,
        )

    history = run_sim(command_fn, duration=4.0)
    roll, pitch, _ = quat_to_euler(history[-1].state.quaternion_bn)
    assert abs(roll - desired_roll) < math.radians(2.0)
    assert abs(pitch) < math.radians(2.0)


def test_rate_loop_respects_actuator_limits() -> None:
    gains = ControllerGains()
    tight_actuators = ActuatorParameters(max_thrust=10.0, max_moment=np.array([0.2, 0.2, 0.1]))
    sim = SixDofSimulator(controller=gains, actuators=tight_actuators, dt=0.002)

    def command_fn(_t: float, _state):
        return PositionVelocityCommand(
            position_ned=np.zeros(3),
            velocity_ned_ff=np.zeros(3),
            accel_ned_ff=np.array([20.0, 0.0, 0.0]),
            yaw_heading=0.0,
        )

    history = sim.run(2.0, command_fn)
    thrusts = [step.controller.thrust_command for step in history]
    assert max(thrusts) <= tight_actuators.max_thrust + 1e-2


def test_yaw_rotation_without_roll_pitch_drift() -> None:
    desired_yaw = math.radians(120.0)

    def command_fn(_t: float, _state):
        return PositionVelocityCommand(
            position_ned=np.zeros(3),
            velocity_ned_ff=np.zeros(3),
            accel_ned_ff=np.zeros(3),
            yaw_heading=desired_yaw,
        )

    history = run_sim(command_fn, duration=3.0)
    roll, pitch, yaw = quat_to_euler(history[-1].state.quaternion_bn)
    assert abs(roll) < math.radians(2.0)
    assert abs(pitch) < math.radians(2.0)
    assert abs(yaw - desired_yaw) < math.radians(2.0)
