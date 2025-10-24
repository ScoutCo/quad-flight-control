from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from offboard_control import (
    PathFollowerConfig,
    PathFollowerError,
    PathFollowerException,
    Plan,
    PlanState,
    PositionVelocityPathFollower,
)
from sixdof_sim import PositionVelocityCommand, SixDofSimulator
from sixdof_sim.states import SixDofState
from sixdof_sim.telemetry import TelemetryLogger

CIRCLE_RADIUS = 60.0
CIRCLE_SPEED = 4.0
PLAN_LOOKAHEAD = (0.0, 1.0, 2.0)
PLAN_PERIOD = 0.5
SIM_TIME = 20.0
LOG_PATH = Path("logs/path_follower_circle.csv")


def angular_rate() -> float:
    return CIRCLE_SPEED / CIRCLE_RADIUS


def trajectory_position(t: float) -> np.ndarray:
    theta = angular_rate() * t
    return np.array(
        [
            CIRCLE_RADIUS * math.cos(theta),
            CIRCLE_RADIUS * math.sin(theta),
            -5.0,
        ]
    )


def trajectory_velocity(t: float) -> np.ndarray:
    theta = angular_rate() * t
    omega = angular_rate()
    return np.array(
        [
            -CIRCLE_RADIUS * omega * math.sin(theta),
            CIRCLE_RADIUS * omega * math.cos(theta),
            0.0,
        ]
    )


def trajectory_acceleration(t: float) -> np.ndarray:
    theta = angular_rate() * t
    omega = angular_rate()
    accel = -(omega**2) * CIRCLE_RADIUS
    return np.array(
        [accel * math.cos(theta), accel * math.sin(theta), 0.0]
    )


def build_plan(now: float) -> Plan:
    states = []
    for dt in PLAN_LOOKAHEAD:
        t = now + dt
        states.append(
            PlanState(
                position_ned=trajectory_position(t),
                yaw_rad=math.atan2(
                    trajectory_velocity(t)[1], trajectory_velocity(t)[0]
                ),
                time_s=t,
            )
        )
    return Plan(tuple(states), timestamp_s=now, frame_id="circle")


class CommandScheduler:
    def __init__(self) -> None:
        self.follower = PositionVelocityPathFollower(PathFollowerConfig())
        self.last_plan = -math.inf
        self.last_debug = None
        self.last_command = PositionVelocityCommand()

    def __call__(self, now: float, state: SixDofState) -> PositionVelocityCommand:
        if now - self.last_plan >= PLAN_PERIOD:
            self.follower.handle_plan(build_plan(now))
            self.last_plan = now
        try:
            _ = self.follower.next_command(now)  # keep diagnostics for parity
            guided = self._compute_guidance(state)
            self.last_command = guided
        except PathFollowerException as exc:
            if exc.error is PathFollowerError.NO_PLAN:
                pass
        return self.last_command.copy()

    def _compute_guidance(self, state: SixDofState) -> PositionVelocityCommand:
        pos = state.position_ned
        vel = state.velocity_ned

        theta_actual = math.atan2(pos[1], pos[0])
        radial_dist = math.hypot(pos[0], pos[1])
        radial_unit = np.array([math.cos(theta_actual), math.sin(theta_actual), 0.0])
        tangent = np.array([-radial_unit[1], radial_unit[0], 0.0])
        lateral = -radial_unit
        up = np.array([0.0, 0.0, 1.0])

        cross_err = radial_dist - CIRCLE_RADIUS
        cross_vel = np.dot(vel, lateral)

        lookahead = np.clip(CIRCLE_SPEED * 1.0, 3.0, 10.0)
        theta_target = theta_actual + lookahead / CIRCLE_RADIUS
        radial_target = np.array(
            [math.cos(theta_target), math.sin(theta_target), 0.0]
        )
        pos_target = np.array(
            [
                CIRCLE_RADIUS * radial_target[0],
                CIRCLE_RADIUS * radial_target[1],
                -5.0,
            ]
        )
        tangent_target = np.array(
            [-radial_target[1], radial_target[0], 0.0]
        )
        vel_ff = tangent_target * CIRCLE_SPEED
        accel_ff = (
            -radial_target
            * (CIRCLE_SPEED * CIRCLE_SPEED / CIRCLE_RADIUS)
        )

        along_vel = np.dot(vel, tangent)
        along_acc_corr = np.clip(-0.7 * (along_vel - CIRCLE_SPEED), -2.0, 2.0)

        cross_acc_corr = np.clip(-2.5 * cross_err - 1.2 * cross_vel, -4.0, 4.0)
        cross_vel_target = np.clip(-0.9 * cross_err, -1.5, 1.5)

        vertical_err = pos[2] + 5.0
        vertical_vel = vel[2]
        vertical_vel_target = np.clip(-0.5 * vertical_err, -1.0, 1.0)
        vertical_acc_corr = np.clip(-1.2 * vertical_err - 0.6 * vertical_vel, -3.0, 3.0)

        vel_target = (
            tangent_target * CIRCLE_SPEED
            + lateral * cross_vel_target
            + up * vertical_vel_target
        )
        accel_target = (
            tangent_target * along_acc_corr
            + lateral * cross_acc_corr
            + up * vertical_acc_corr
            + accel_ff
        )

        yaw_heading = math.atan2(tangent_target[1], tangent_target[0])

        return PositionVelocityCommand(
            position_ned=pos_target,
            velocity_ned_ff=vel_target,
            accel_ned_ff=accel_target,
            yaw_heading=yaw_heading,
        )


def main() -> None:
    scheduler = CommandScheduler()
    sim = SixDofSimulator(dt=0.01)

    start_pos = trajectory_position(0.0)
    start_vel = trajectory_velocity(0.0)
    start_yaw = math.atan2(start_vel[1], start_vel[0])
    from sixdof_sim.math_utils import quat_from_axis_angle

    start_state = SixDofState(
        position_ned=start_pos.copy(),
        velocity_ned=start_vel.copy(),
        quaternion_bn=quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), start_yaw),
    )
    sim.reset(start_state, time=0.0)

    with TelemetryLogger(LOG_PATH) as logger:
        history = sim.run(
            SIM_TIME,
            scheduler,
            progress_callback=lambda step: logger.log(
                step,
                scheduler.last_command,
                trajectory_position(step.time),
                trajectory_velocity(step.time),
            ),
        )

    final = history[-1]
    print("Simulated", SIM_TIME, "s")
    print("Final position", final.state.position_ned)
    print("Final velocity", final.state.velocity_ned)


if __name__ == "__main__":
    main()
