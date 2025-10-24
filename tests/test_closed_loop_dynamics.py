"""Integration tests using the full dynamics + actuator + controller stack."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from sixdof_sim import PositionVelocityCommand, SixDofSimulator


def run_sim(command_fn, duration: float = 5.0, dt: float = 0.01):
    sim = SixDofSimulator(dt=dt)
    history = sim.run(duration, command_fn)
    return history


def test_constant_velocity_holds_altitude() -> None:
    target_vel = np.array([1.0, 0.0, 0.0])

    def command_fn(_t: float, state) -> PositionVelocityCommand:
        return PositionVelocityCommand(
            position_ned=state.position_ned.copy(),
            velocity_ned_ff=target_vel,
            accel_ned_ff=np.zeros(3),
            yaw_heading=0.0,
        )

    hist = run_sim(command_fn, duration=5.0)
    final = hist[-1].state
    assert abs(final.position_ned[2]) < 0.5
    assert abs(final.velocity_ned[0] - target_vel[0]) < 0.5


def test_centripetal_feedforward_tracks() -> None:
    radius = 40.0
    speed = 1.5
    omega = speed / radius

    def command_fn(t: float, state) -> PositionVelocityCommand:
        theta = omega * t
        vel_ff = np.array([
            -radius * omega * math.sin(theta),
            radius * omega * math.cos(theta),
            0.0,
        ])
        acc_ff = np.array([
            -radius * omega * omega * math.cos(theta),
            -radius * omega * omega * math.sin(theta),
            0.0,
        ])
        return PositionVelocityCommand(
            position_ned=state.position_ned.copy(),
            velocity_ned_ff=vel_ff,
            accel_ned_ff=acc_ff,
            yaw_heading=theta,
        )

    dt = 0.01
    hist = run_sim(command_fn, duration=5.0, dt=dt)
    final = hist[-1].state
    assert abs(final.position_ned[2]) < 1.0
    v_prev = hist[-2].state.velocity_ned
    v_curr = final.velocity_ned
    measured_acc = (v_curr - v_prev) / dt
    expected_acc_mag = speed * speed / radius
    assert math.isclose(
        math.hypot(measured_acc[0], measured_acc[1]),
        expected_acc_mag,
        rel_tol=0.2,
    )
