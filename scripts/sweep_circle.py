from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from sixdof_sim import PositionVelocityCommand, SixDofSimulator
from sixdof_sim.math_utils import quat_to_euler
from sixdof_sim.states import SixDofState


@dataclass
class SweepResult:
    radius: float
    speed: float
    max_position_error: float
    max_altitude_error: float
    max_thrust: float
    max_tilt_deg: float

    def __str__(self) -> str:
        return (
            f"r={self.radius:5.1f} m, v={self.speed:4.1f} m/s, "
            f"max_xy_err={self.max_position_error:6.2f} m, max_z_err={self.max_altitude_error:6.2f} m, "
            f"max_thrust={self.max_thrust:6.1f} N, max_tilt={self.max_tilt_deg:5.2f} deg"
        )


def desired_state(radius: float, speed: float, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    omega = speed / radius
    theta = omega * t
    position = np.array([
        radius * math.cos(theta),
        radius * math.sin(theta),
        -5.0,
    ])
    velocity = np.array([
        -radius * omega * math.sin(theta),
        radius * omega * math.cos(theta),
        0.0,
    ])
    acceleration = np.array([
        -radius * omega * omega * math.cos(theta),
        -radius * omega * omega * math.sin(theta),
        0.0,
    ])
    yaw = math.atan2(velocity[1], velocity[0])
    return position, velocity, acceleration, yaw


def run_case(radius: float, speed: float, duration: float = 15.0, dt: float = 0.01) -> SweepResult:
    sim = SixDofSimulator(dt=dt)
    start_pos, start_vel, _, _ = desired_state(radius, speed, 0.0)
    start_state = SixDofState(position_ned=start_pos.copy(), velocity_ned=start_vel.copy())
    sim.reset(start_state, time=0.0)

    def command_fn(t: float, _state) -> PositionVelocityCommand:
        pos, vel, accel, yaw = desired_state(radius, speed, t)
        return PositionVelocityCommand(
            position_ned=pos,
            velocity_ned_ff=vel,
            accel_ned_ff=accel,
            yaw_heading=yaw,
        )

    history = sim.run(duration, command_fn)

    max_xy_err = 0.0
    max_z_err = 0.0
    max_thrust = 0.0
    max_tilt = 0.0

    for step in history:
        t = step.time
        desired_pos, _, _, _ = desired_state(radius, speed, t)
        pos_err = step.state.position_ned - desired_pos
        max_xy_err = max(max_xy_err, np.linalg.norm(pos_err[:2]))
        max_z_err = max(max_z_err, abs(pos_err[2]))
        max_thrust = max(max_thrust, step.controller.thrust_command)
        roll, pitch, _ = quat_to_euler(step.state.quaternion_bn)
        max_tilt = max(max_tilt, math.degrees(max(abs(roll), abs(pitch))))

    return SweepResult(radius, speed, max_xy_err, max_z_err, max_thrust, max_tilt)


def main() -> None:
    radii = [40.0, 60.0, 80.0]
    speeds = [0.5, 1.0, 2.0, 3.0, 4.0]
    results = []
    for r in radii:
        for s in speeds:
            res = run_case(r, s)
            results.append(res)
            print(res)

    out_path = Path("logs/circle_sweep.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for res in results:
            f.write(str(res) + "\n")


if __name__ == "__main__":
    main()
