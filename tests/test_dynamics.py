"""Sanity checks for the low-level dynamics model."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from sixdof_sim.dynamics import RigidBodyDynamics
from sixdof_sim.parameters import EnvironmentParameters, VehicleParameters
from sixdof_sim.states import SixDofState
from sixdof_sim.math_utils import quat_from_axis_angle


def run_rk4(dynamics: RigidBodyDynamics, state: SixDofState, thrust: float, dt: float, steps: int) -> SixDofState:
    s = state.copy()
    env_params = EnvironmentParameters()
    for _ in range(steps):
        s = dynamics.rk4_step(
            s,
            thrust=thrust,
            moment_body=np.zeros(3),
            env_gravity=np.array([0.0, 0.0, env_params.gravity]),
            wind_ned=np.zeros(3),
            external_force_body=np.zeros(3),
            external_moment_body=np.zeros(3),
            dt=dt,
        )
    return s


def test_free_fall_gravity() -> None:
    params = VehicleParameters()
    dynamics = RigidBodyDynamics(params)
    state0 = SixDofState(position_ned=np.zeros(3), velocity_ned=np.zeros(3))
    dt = 0.01
    duration = 1.0
    steps = int(duration / dt)
    state = run_rk4(dynamics, state0, thrust=0.0, dt=dt, steps=steps)

    expected_z = 0.5 * 9.80665 * duration * duration
    expected_vz = 9.80665 * duration
    assert math.isclose(state.position_ned[2], expected_z, rel_tol=1e-3)
    assert math.isclose(state.velocity_ned[2], expected_vz, rel_tol=1e-3)


def test_constant_thrust_acc() -> None:
    params = VehicleParameters()
    dynamics = RigidBodyDynamics(params)
    state0 = SixDofState(position_ned=np.zeros(3), velocity_ned=np.zeros(3))
    thrust = params.mass * 12.0
    dt = 0.01
    duration = 1.0
    steps = int(duration / dt)
    state = run_rk4(dynamics, state0, thrust=thrust, dt=dt, steps=steps)

    net_acc = thrust / params.mass - 9.80665
    expected_vz = net_acc * duration
    expected_z = 0.5 * net_acc * duration * duration
    assert math.isclose(state.velocity_ned[2], expected_vz, rel_tol=1e-3)
    assert math.isclose(state.position_ned[2], expected_z, rel_tol=1e-3)


def test_tilt_generates_lateral_accel() -> None:
    params = VehicleParameters()
    dynamics = RigidBodyDynamics(params)

    roll_rad = math.radians(10.0)
    quat = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), roll_rad)
    state0 = SixDofState(
        position_ned=np.zeros(3),
        velocity_ned=np.zeros(3),
        quaternion_bn=quat,
        angular_velocity_body=np.zeros(3),
    )

    thrust = params.mass * 9.80665
    dt = 0.01
    steps = int(0.5 / dt)
    state = run_rk4(dynamics, state0, thrust=thrust, dt=dt, steps=steps)

    expected_ax = 9.80665 * math.tan(roll_rad)
    measured_ax = (state.velocity_ned[0]) / (steps * dt)
    assert math.isclose(measured_ax, expected_ax, rel_tol=5e-2)
