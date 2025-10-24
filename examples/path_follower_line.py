from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from sixdof_sim import PositionVelocityCommand, SixDofSimulator
from sixdof_sim.telemetry import TelemetryLogger
from sixdof_sim.states import SixDofState

LOG_PATH = Path("logs/path_follower_line.csv")
SIM_TIME = 20.0


def desired_state(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    pos = np.array([0.1 * t, 0.0, -5.0])
    vel = np.array([0.1, 0.0, 0.0])
    acc = np.zeros(3)
    yaw = 0.0
    return pos, vel, acc, yaw


def command_fn(t: float, _state: SixDofState) -> PositionVelocityCommand:
    pos, vel, acc, yaw = desired_state(t)
    return PositionVelocityCommand(position_ned=pos, velocity_ned_ff=vel, accel_ned_ff=acc, yaw_heading=yaw)


def main() -> None:
    sim = SixDofSimulator(dt=0.01)
    with TelemetryLogger(LOG_PATH) as logger:
        hist = sim.run(
            SIM_TIME,
            command_fn,
            progress_callback=lambda step: logger.log(
                step,
                command_fn(step.time, step.state),
                *desired_state(step.time)[:2]
            ),
        )
    final = hist[-1]
    print("Simulated", SIM_TIME, "s")
    print("Final position", final.state.position_ned)
    print("Final velocity", final.state.velocity_ned)


if __name__ == "__main__":
    main()
