from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from offboard_control import (
    PathFollowerConfig,
    PathFollowerDebug,
    PathFollowerException,
    Plan,
    PlanState,
    PositionVelocityPathFollower,
)
from sixdof_sim import PositionVelocityCommand, SixDofSimulator
from sixdof_sim.states import SixDofState
from sixdof_sim.telemetry import TelemetryLogger

PLAN_LOOKAHEAD_OFFSETS = (1.0, 2.0, 3.0)
PLAN_UPDATE_PERIOD = 2.0  # 0.5 Hz planner
PLANNER_DELAY_S = 1.3
SIM_FINAL_TIME = 30.0
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_FILE = LOG_DIR / "path_follower_sinusoid.csv"

FORWARD_SPEED = 3.0  # m/s along x
Y_AMPLITUDE = 2.0  # m
Y_FREQUENCY = 0.2  # rad/s
Z_BASE = -5.0  # m (NED down)
Z_AMPLITUDE = 0.8  # m
Z_FREQUENCY = 0.3  # rad/s


def trajectory_position(time_s: float) -> np.ndarray:
    x = FORWARD_SPEED * time_s
    y = Y_AMPLITUDE * math.sin(Y_FREQUENCY * time_s)
    z = Z_BASE + Z_AMPLITUDE * math.sin(Z_FREQUENCY * time_s)
    return np.array([x, y, z], dtype=float)


def trajectory_velocity(time_s: float) -> np.ndarray:
    dx = FORWARD_SPEED
    dy = Y_AMPLITUDE * Y_FREQUENCY * math.cos(Y_FREQUENCY * time_s)
    dz = Z_AMPLITUDE * Z_FREQUENCY * math.cos(Z_FREQUENCY * time_s)
    return np.array([dx, dy, dz], dtype=float)


def trajectory_yaw(time_s: float) -> float:
    vel = trajectory_velocity(time_s)
    return float(math.atan2(vel[1], vel[0]))


def build_plan(now_s: float) -> tuple[Plan, float]:
    plan_timestamp = now_s - PLANNER_DELAY_S
    states: Iterable[PlanState] = (
        PlanState(
            position_ned=trajectory_position(plan_timestamp + offset),
            yaw_rad=trajectory_yaw(plan_timestamp + offset),
            time_s=plan_timestamp + offset,
        )
        for offset in PLAN_LOOKAHEAD_OFFSETS
    )
    return Plan(tuple(states), timestamp_s=plan_timestamp, frame_id="ref_ned"), plan_timestamp


@dataclass
class CommandGenerator:
    follower: PositionVelocityPathFollower
    last_plan_time: float = -math.inf
    last_command: PositionVelocityCommand = field(
        default_factory=PositionVelocityCommand
    )
    last_debug: PathFollowerDebug | None = None
    last_plan_timestamp: float = float("nan")

    def __call__(self, time_s: float, state: SixDofState) -> PositionVelocityCommand:
        if (time_s - self.last_plan_time) >= PLAN_UPDATE_PERIOD - 1e-9:
            plan, plan_timestamp = build_plan(time_s)
            self.follower.handle_plan(plan)
            self.last_plan_time = time_s
            self.last_plan_timestamp = plan_timestamp
        try:
            result = self.follower.next_command(time_s)
            self.last_command = result.command
            self.last_debug = result.debug
        except PathFollowerException:
            hold_position = state.position_ned.copy()
            self.last_command = PositionVelocityCommand(
                position_ned=hold_position, velocity_ned_ff=np.zeros(3), yaw_heading=0.0
            )
            self.last_debug = None
        return self.last_command


def analyze_results(history: list, log_path: Path) -> None:
    if not history:
        print("No simulation history recorded.")
        return

    errors = []
    for step in history:
        ref_pos = trajectory_position(step.time)
        error = step.state.position_ned - ref_pos
        errors.append(np.linalg.norm(error))

    rms_error = math.sqrt(float(np.mean(np.square(errors))))
    max_error = float(np.max(errors))
    final_state = history[-1].state

    print(f"Simulated {len(history)} steps over {history[-1].time:.2f} s.")
    print(f"Final position NED: {final_state.position_ned}")
    print(f"Final velocity NED: {final_state.velocity_ned}")
    print(f"RMS position error: {rms_error:.3f} m")
    print(f"Max position error: {max_error:.3f} m")
    print(f"Telemetry log written to: {log_path}")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    follower = PositionVelocityPathFollower(
        PathFollowerConfig(max_plan_age_s=5.0, lookahead_offset_s=0.5)
    )
    command_generator = CommandGenerator(follower=follower)

    sim = SixDofSimulator(dt=0.01)

    with TelemetryLogger(LOG_FILE) as logger:

        def log_step(step_result) -> None:
            command = command_generator.last_command
            debug = command_generator.last_debug
            interp_state = debug.interpolated_state if debug else None

            logger.log(
                step_result,
                command,
                plan_timestamp=command_generator.last_plan_timestamp
                if math.isfinite(command_generator.last_plan_timestamp)
                else None,
                reference_position=trajectory_position(step_result.time),
                reference_velocity=trajectory_velocity(step_result.time),
                reference_yaw=trajectory_yaw(step_result.time),
                feedforward_velocity=(
                    debug.feedforward_velocity_ned if debug else None
                ),
                interpolated_position=(
                    interp_state.position_ned if interp_state is not None else None
                ),
                interpolated_yaw=interp_state.yaw_rad if interp_state else None,
                controller_state=step_result.controller,
                actuator_state=step_result.actuators,
            )

        history = sim.run(
            final_time=SIM_FINAL_TIME,
            command_fn=command_generator,
            progress_callback=log_step,
        )

    analyze_results(history, LOG_FILE)


if __name__ == "__main__":
    main()
