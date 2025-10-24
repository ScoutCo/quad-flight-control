from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from path_follower import (
    DEFAULT_PLAN_LOOKAHEAD_OFFSETS,
    PathFollowerConfig,
    PositionVelocityPathFollower,
    SimCommandGenerator,
)
from sim import (
    SimulationStep,
    Simulator,
    SimulatorConfig,
    TelemetryLogger,
)

from common import Trajectory


def analyze_history(history: Iterable[SimulationStep], trajectory: Trajectory) -> None:
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
    simulator_config: SimulatorConfig | None = None,
    follower_config: PathFollowerConfig | None = None,
    plan_offsets_s: Sequence[float] = DEFAULT_PLAN_LOOKAHEAD_OFFSETS,
    planner_delay_s: float = 1.3,
    plan_update_period_s: float = 1.0,
) -> list[SimulationStep]:
    if follower_config is None:
        follower_config = PathFollowerConfig(
            lookahead_offset_s=0.5, max_plan_age_s=max(5.0, plan_update_period_s + planner_delay_s + 1.0)
        )

    follower = PositionVelocityPathFollower(follower_config)
    command_generator = SimCommandGenerator(
        follower=follower,
        trajectory=trajectory,
        plan_offsets_s=plan_offsets_s,
        planner_delay_s=planner_delay_s,
        plan_update_period_s=plan_update_period_s,
    )

    simulator = Simulator(simulator_config)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[SimulationStep]
    with TelemetryLogger(log_path) as logger:

        def log_step(step: SimulationStep) -> None:
            command = command_generator.last_command
            if command is not None:
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
