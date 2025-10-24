"""Integration tests for the path follower using the simulator."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from path_follower.config import PathFollowerConfig
from path_follower.path_follower import PathFollowerException, PositionVelocityPathFollower
from path_follower.plan import Plan, PlanState
from sim import Simulator, SimulatorConfig, TelemetryLogger


def _build_line_plan() -> Plan:
    states = (
        PlanState(position_ned=np.array([0.0, 0.0, -1.0]), yaw_rad=0.0, time_s=0.0),
        PlanState(position_ned=np.array([5.0, 0.0, -1.0]), yaw_rad=0.0, time_s=5.0),
        PlanState(position_ned=np.array([10.0, 5.0, -2.0]), yaw_rad=math.radians(45.0), time_s=11.0),
        PlanState(position_ned=np.array([10.0, 5.0, -2.0]), yaw_rad=math.radians(45.0), time_s=13.0),
    )
    return Plan(states=states, timestamp_s=0.0, frame_id="test_path")


def test_path_follower_tracks_plan(tmp_path) -> None:
    dt = 0.02
    log_path = tmp_path / "sim_follower.csv"
    sim = Simulator(SimulatorConfig(dt=dt))
    follower = PositionVelocityPathFollower(
        PathFollowerConfig(max_plan_age_s=50.0, lookahead_offset_s=0.5)
    )
    follower.handle_plan(_build_line_plan())

    duration = 11.5
    steps = int(duration / dt)
    history_positions = []

    with TelemetryLogger(log_path) as logger:
        for _ in range(steps):
            command = follower.next_command(sim.time_s).command
            step = sim.step(command, dt=dt)
            history_positions.append(step.state.position_ned.copy())
            logger.log(
                step,
                command,
                reference_position=command.position_ned,
                reference_velocity=command.velocity_ned_ff,
            )

    final_state = sim.state
    assert np.allclose(final_state.position_ned, np.array([10.0, 5.0, -2.0]), atol=0.6)
    assert np.linalg.norm(final_state.velocity_ned[:2]) < 0.7

    path_length = sum(
        np.linalg.norm(b - a)
        for a, b in zip(history_positions[:-1], history_positions[1:])
    )
    straight_line = np.linalg.norm(history_positions[-1] - history_positions[0])
    assert path_length >= straight_line
    assert path_length < 1.5 * straight_line
    assert log_path.exists() and log_path.stat().st_size > 0


def test_path_follower_respects_plan_start_time() -> None:
    dt = 0.02
    sim = Simulator(SimulatorConfig(dt=dt))
    follower = PositionVelocityPathFollower(
        PathFollowerConfig(max_plan_age_s=50.0, lookahead_offset_s=0.5)
    )

    plan = _build_line_plan()
    follower.handle_plan(plan)

    with pytest.raises(PathFollowerException):
        follower.next_command(plan.start_time_s - 10.0)

    command = follower.next_command(sim.time_s).command
    assert np.allclose(command.position_ned, plan.states[0].position_ned)
    step = sim.step(command, dt=dt)
    assert step.state.position_ned[2] < 0.0
