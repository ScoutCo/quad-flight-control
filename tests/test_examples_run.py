import math
from collections.abc import Callable

import numpy as np
import pytest

from examples import (
    circle_trajectory,
    line_trajectory,
    run_path_follower_example,
    sinusoid_trajectory,
    zigzag_trajectory,
)
from common import Trajectory
from sim.math_utils import quat_to_euler


TrajectoryFactory = Callable[[], Trajectory]


def _yaw_from_quaternion(quat: np.ndarray) -> float:
    return float(quat_to_euler(quat)[2])


def _trajectory_cases() -> list[tuple[str, TrajectoryFactory, float]]:
    return [
        (
            "line",
            lambda: line_trajectory(
                speed_m_s=4.5,
                heading_rad=0.1,
                altitude_m=-2.0,
                lateral_offset_m=0.0,
            ),
            25.0,
        ),
        (
            "circle",
            lambda: circle_trajectory(
                radius_m=10.0,
                altitude_m=-5.0,
                period_s=60.0,
                phase_rad=0.0,
            ),
            32.0,
        ),
        (
            "sinusoid",
            lambda: sinusoid_trajectory(
                forward_speed_m_s=5.0,
                y_amplitude_m=5.0,
                y_frequency_hz=0.05,
                z_amplitude_m=1.0,
                z_frequency_hz=0.025,
            ),
            30.0,
        ),
        (
            "zigzag",
            lambda: zigzag_trajectory(
                anchor_ned=(0.0, 0.0, -3.0),
                heading_deg=0.0,
                segment_length_m=40.0,
                num_segments=8,
                offset_angle_deg=40.0,
                speed_m_s=5.0,
                start_with_positive_offset=True,
            ),
            70.0,
        ),
    ]


@pytest.mark.parametrize(
    ("name", "trajectory_factory", "duration_s"),
    _trajectory_cases(),
)
def test_examples_track_reference(name: str, trajectory_factory: TrajectoryFactory, duration_s: float, tmp_path) -> None:
    trajectory = trajectory_factory()
    log_path = tmp_path / f"sim_{name}.csv"
    history = run_path_follower_example(
        trajectory=trajectory,
        log_path=log_path,
        final_time_s=duration_s,
    )

    assert history, f"{name} example produced no simulation steps"
    assert log_path.exists(), f"{name} log was not written"

    pos_errors: list[float] = []
    vel_errors: list[float] = []
    yaw_errors: list[float] = []

    settle_time_s = 5.0

    for step in history:
        if step.time_s < settle_time_s:
            continue

        ref_pos = trajectory.position(step.time_s)
        ref_vel = trajectory.velocity(step.time_s)
        ref_yaw = trajectory.yaw(step.time_s)

        pos_errors.append(float(np.linalg.norm(step.state.position_ned - ref_pos)))
        vel_errors.append(float(np.linalg.norm(step.state.velocity_ned - ref_vel)))

        actual_yaw = _yaw_from_quaternion(step.state.quaternion_bn)
        yaw_errors.append(abs(math.remainder(actual_yaw - ref_yaw, 2.0 * math.pi)))

    assert pos_errors, "No samples remained after settling period"

    pos_arr = np.asarray(pos_errors)
    vel_arr = np.asarray(vel_errors)
    yaw_arr = np.asarray(yaw_errors)

    pos_rms = float(np.sqrt(np.mean(pos_arr**2)))
    pos_p95 = float(np.percentile(pos_arr, 95))
    pos_max = float(pos_arr.max())

    vel_rms = float(np.sqrt(np.mean(vel_arr**2)))
    vel_p95 = float(np.percentile(vel_arr, 95))
    vel_max = float(vel_arr.max())

    yaw_rms = float(np.sqrt(np.mean(yaw_arr**2)))
    yaw_p95 = float(np.percentile(yaw_arr, 95))
    yaw_max = float(yaw_arr.max())

    # Allow reasonable tracking error margins while still flagging regressions.
    assert pos_rms < 1.5, f"{name} position RMS too high: {pos_rms:.3f} m"
    assert pos_p95 < 3.5, f"{name} position 95th percentile too high: {pos_p95:.3f} m"
    assert pos_max < 6.0, f"{name} position max too high: {pos_max:.3f} m"

    assert vel_rms < 1.5, f"{name} velocity RMS too high: {vel_rms:.3f} m/s"
    assert vel_p95 < 3.5, f"{name} velocity 95th percentile too high: {vel_p95:.3f} m/s"
    assert vel_max < 5.5, f"{name} velocity max too high: {vel_max:.3f} m/s"

    assert yaw_rms < 0.6, f"{name} yaw RMS too high: {math.degrees(yaw_rms):.2f} deg"
    assert yaw_p95 < 1.0, f"{name} yaw 95th percentile too high: {math.degrees(yaw_p95):.2f} deg"
    assert yaw_max < 1.5, f"{name} yaw max too high: {math.degrees(yaw_max):.2f} deg"
