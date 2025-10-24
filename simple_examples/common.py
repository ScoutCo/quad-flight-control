from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from offboard_control.config import PathFollowerConfig
from offboard_control.path_follower import (
    PathFollowerDebug,
    PathFollowerException,
    Plan,
    PlanState,
    PositionVelocityPathFollower,
)
from simple_sim import (
    SimpleSimulationStep,
    SimpleSimulator,
    SimpleSimulatorConfig,
    SimpleState,
    SimpleTelemetryLogger,
)
from sixdof_sim.commands import PositionVelocityCommand

from .trajectories import Trajectory


# Extend the plan far enough that lookahead checks remain valid between planner updates.
DEFAULT_PLAN_LOOKAHEAD_OFFSETS = (1.0, 2.0, 4.5)


def _build_plan(
    trajectory: Trajectory,
    plan_offsets_s: Sequence[float],
    now_s: float,
    planner_delay_s: float,
) -> tuple[Plan, float]:
    plan_timestamp = now_s - planner_delay_s
    states: tuple[PlanState, ...] = tuple(
        PlanState(
            position_ned=trajectory.position(plan_timestamp + offset),
            yaw_rad=trajectory.yaw(plan_timestamp + offset),
            time_s=plan_timestamp + offset,
        )
        for offset in plan_offsets_s
    )
    if len(states) < 2:
        raise ValueError("Path follower requires at least two plan states.")
    return Plan(states=states, timestamp_s=plan_timestamp, frame_id="ref_ned"), plan_timestamp


@dataclass
class CommandGenerator:
    follower: PositionVelocityPathFollower
    trajectory: Trajectory
    plan_offsets_s: Sequence[float]
    planner_delay_s: float
    plan_update_period_s: float
    last_plan_time_s: float = -math.inf
    last_command: PositionVelocityCommand = field(default_factory=PositionVelocityCommand)
    last_debug: PathFollowerDebug | None = None
    last_plan_timestamp_s: float = float("nan")

    def __call__(self, time_s: float, state: SimpleState) -> PositionVelocityCommand:
        if (time_s - self.last_plan_time_s) >= self.plan_update_period_s - 1e-9:
            plan, plan_timestamp = _build_plan(
                self.trajectory, self.plan_offsets_s, time_s, self.planner_delay_s
            )
            self.follower.handle_plan(plan)
            self.last_plan_time_s = time_s
            self.last_plan_timestamp_s = plan_timestamp

        try:
            result = self.follower.next_command(time_s)
        except PathFollowerException:
            hold_position = state.position_ned.copy()
            self.last_command = PositionVelocityCommand(
                position_ned=hold_position,
                velocity_ned_ff=np.zeros(3),
                yaw_heading=float(self.trajectory.yaw(time_s)),
            )
            self.last_debug = None
            return self.last_command

        self.last_command = result.command
        self.last_debug = result.debug
        return self.last_command


def analyze_history(history: Iterable[SimpleSimulationStep], trajectory: Trajectory) -> None:
    history = list(history)
    if not history:
        print("No simulation history recorded.")
        return

    position_errors: list[float] = []
    for step in history:
        ref_pos = trajectory.position(step.time_s)
        error = step.state.position_ned - ref_pos
        position_errors.append(float(np.linalg.norm(error)))

    rms_error = math.sqrt(float(np.mean(np.square(position_errors))))
    max_error = float(np.max(position_errors))
    final_state = history[-1].state

    print(f"Simulated {len(history)} steps over {history[-1].time_s:.2f} s.")
    print(f"Final position NED: {np.array2string(final_state.position_ned, precision=3)}")
    print(f"Final velocity NED: {np.array2string(final_state.velocity_ned, precision=3)}")
    print(f"RMS position error: {rms_error:.3f} m")
    print(f"Max position error: {max_error:.3f} m")


def run_path_follower_example(
    trajectory: Trajectory,
    log_path: Path,
    final_time_s: float = 30.0,
    *,
    simulator_config: SimpleSimulatorConfig | None = None,
    follower_config: PathFollowerConfig | None = None,
    plan_offsets_s: Sequence[float] = DEFAULT_PLAN_LOOKAHEAD_OFFSETS,
    planner_delay_s: float = 1.3,
    plan_update_period_s: float = 1.0,
) -> list[SimpleSimulationStep]:
    if follower_config is None:
        follower_config = PathFollowerConfig(
            lookahead_offset_s=0.5, max_plan_age_s=max(5.0, plan_update_period_s + planner_delay_s + 1.0)
        )

    follower = PositionVelocityPathFollower(follower_config)
    command_generator = CommandGenerator(
        follower=follower,
        trajectory=trajectory,
        plan_offsets_s=plan_offsets_s,
        planner_delay_s=planner_delay_s,
        plan_update_period_s=plan_update_period_s,
    )

    simulator = SimpleSimulator(simulator_config)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[SimpleSimulationStep]
    with SimpleTelemetryLogger(log_path) as logger:

        def log_step(step: SimpleSimulationStep) -> None:
            command = command_generator.last_command
            logger.log(
                step,
                command,
                reference_position=trajectory.position(step.time_s),
                reference_velocity=trajectory.velocity(step.time_s),
            )

        history = simulator.run(
            final_time_s=final_time_s,
            command_fn=command_generator,
            progress_callback=log_step,
        )

    analyze_history(history, trajectory)
    print(f"Telemetry log written to: {log_path}")
    return history
