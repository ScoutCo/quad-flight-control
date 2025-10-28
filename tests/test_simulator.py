"""Unit tests for the simulator dynamics."""

import math
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from quad_flight_control.sim import (
    PositionVelocityCommand,
    Simulator,
    SimulatorConfig,
    TelemetryLogger,
)
from quad_flight_control.sim import quat_to_euler
from quad_flight_control.sim.math_utils import quat_to_rotation_matrix, rotation_matrix_to_quat


def test_constant_velocity_command_reached() -> None:
    config = SimulatorConfig(dt=0.02)
    sim = Simulator(config)

    velocity_cmd = np.array([2.0, 0.0, 0.0])
    steps = int(3.0 / config.dt)
    for _ in range(steps):
        target_position = velocity_cmd * (sim.time_s + config.dt)
        command = PositionVelocityCommand(
            position_ned=target_position,
            velocity_ned_ff=velocity_cmd,
            accel_ned_ff=np.zeros(3),
            yaw_heading=0.0,
        )
        sim.step(command)

    assert sim.state.velocity_ned[0] == pytest.approx(2.0, abs=0.1)
    assert np.linalg.norm(sim.state.velocity_ned[1:]) < 0.1


def test_position_hold_converges() -> None:
    config = SimulatorConfig(dt=0.02)
    sim = Simulator(config)
    target = np.array([5.0, -3.0, -2.0])

    command = PositionVelocityCommand(
        position_ned=target,
        velocity_ned_ff=np.zeros(3),
        accel_ned_ff=np.zeros(3),
        yaw_heading=0.0,
    )

    for _ in range(int(8.0 / config.dt)):
        sim.step(command)

    assert np.allclose(sim.state.position_ned, target, atol=0.3)
    assert np.linalg.norm(sim.state.velocity_ned) < 0.3


def test_attitude_reflects_lateral_motion() -> None:
    config = SimulatorConfig(dt=0.02)
    sim = Simulator(config)

    target = np.array([0.0, 5.0, -1.0])
    command = PositionVelocityCommand(
        position_ned=target,
        velocity_ned_ff=np.zeros(3),
        accel_ned_ff=np.zeros(3),
        yaw_heading=0.0,
    )
    for _ in range(int(4.0 / config.dt)):
        sim.step(command)

    roll, pitch, yaw = quat_to_euler(sim.state.quaternion_bn)
    assert abs(roll) > math.radians(0.5)
    assert yaw == pytest.approx(0.0, abs=math.radians(2.0))


@pytest.mark.parametrize('ref_velocity', [np.zeros(3), np.array([0.5, -0.2, 0.0])])
def test_telemetry_logger_records_data(tmp_path, ref_velocity) -> None:
    config = SimulatorConfig(dt=0.05)
    sim = Simulator(config)
    log_path = tmp_path / 'sim_log.csv'

    command = PositionVelocityCommand(
        position_ned=np.array([1.0, -0.5, -0.2]),
        velocity_ned_ff=np.array([0.3, 0.1, -0.05]),
        accel_ned_ff=np.array([0.0, 0.0, 0.1]),
        yaw_heading=0.2,
    )

    with TelemetryLogger(log_path) as logger:
        step = sim.step(command)
        logger.log(step, command, reference_position=command.position_ned, reference_velocity=ref_velocity)

    assert log_path.exists()
    with log_path.open() as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 2
    headers = lines[0].split(',')
    expected_headers = TelemetryLogger.HEADERS
    assert headers == expected_headers
    values = [float(x) for x in lines[1].split(',')]
    assert math.isclose(values[0], config.dt, rel_tol=1e-6)
    assert np.allclose(values[7:10], ref_velocity.tolist(), atol=1e-6)
    assert values[4] != 0.0 or values[5] != 0.0 or values[6] != 0.0
    assert values[22] != 0.0 or values[23] != 0.0 or values[24] != 0.0


def test_tilt_limit_enforced() -> None:
    config = SimulatorConfig(max_tilt_deg=35.0)
    sim = Simulator(config)

    yaw = 0.0
    thrust_dir = np.array([1.0, 0.0, 0.1], dtype=float)
    thrust_dir = thrust_dir / np.linalg.norm(thrust_dir)

    x_ref = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=float)
    if np.linalg.norm(x_ref) < 1e-6:
        x_ref = np.array([1.0, 0.0, 0.0], dtype=float)

    y_body = np.cross(thrust_dir, x_ref)
    if np.linalg.norm(y_body) < 1e-6:
        x_ref = np.array([0.0, 1.0, 0.0], dtype=float)
        y_body = np.cross(thrust_dir, x_ref)
    y_body = y_body / np.linalg.norm(y_body)
    x_body = np.cross(y_body, thrust_dir)
    x_body = x_body / np.linalg.norm(x_body)

    rot_bn = np.column_stack((x_body, y_body, thrust_dir))
    sim.state.quaternion_bn = rotation_matrix_to_quat(rot_bn)

    sim._enforce_tilt_limit()

    limited_rot = quat_to_rotation_matrix(sim.state.quaternion_bn)
    limited_z = limited_rot[:, 2]
    tilt_deg = math.degrees(math.acos(np.clip(limited_z[2], -1.0, 1.0)))

    assert tilt_deg <= config.max_tilt_deg + 1e-6


def test_yaw_rate_limit() -> None:
    max_rate_deg_s = 45.0
    config = SimulatorConfig(dt=0.02, max_yaw_rate_rad_s=math.radians(max_rate_deg_s))
    sim = Simulator(config)

    command = PositionVelocityCommand(yaw_heading=math.pi)
    sim.step(command)

    expected_delta = math.radians(max_rate_deg_s) * config.dt
    assert math.isclose(sim._target_yaw, expected_delta, rel_tol=1e-6, abs_tol=1e-6)
