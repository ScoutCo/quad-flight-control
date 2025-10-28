#!/usr/bin/env python3

"""Fly a reference trajectory using MAVSDK offboard/Guided control.

This script mirrors the simulator examples but streams position+velocity
setpoints to a connected autopilot (PX4 offboard or ArduPilot guided).

Example usage:
  python scripts/path_follow_flight.py --connection udp://:14550

Select the reference trajectory inside ``build_reference_trajectory`` below.
"""

import argparse
import asyncio
import csv
import math
import sys
import time
from dataclasses import dataclass, field
import datetime
from pathlib import Path
from typing import IO

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from mavsdk import System
from mavsdk.action import ActionError
from mavsdk.offboard import OffboardError, PositionNedYaw, VelocityNedYaw

from common import (
    Trajectory,
    zigzag_trajectory,
    line_trajectory,
    circle_trajectory,
    sinusoid_trajectory,
)
from path_follower import (
    DEFAULT_PLAN_LOOKAHEAD_OFFSETS,
    PathFollowerConfig,
    PathFollowerException,
    PositionVelocityPathFollower,
    build_plan,
)


COMMAND_RATE_HZ_DEFAULT = 25.0
PLAN_UPDATE_PERIOD_S_DEFAULT = 1.0
PLANNER_DELAY_S_DEFAULT = 1.3


@dataclass
class TelemetryCache:
    position_ned: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    velocity_ned: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    yaw_rad: float = float("nan")
    position_timestamp: float = float("nan")
    yaw_timestamp: float = float("nan")


def offset_trajectory(base: Trajectory, anchor_ned: np.ndarray) -> Trajectory:
    """Return a trajectory translated so that its origin sits at anchor_ned."""

    anchor = anchor_ned.astype(float)

    def position(time_s: float) -> np.ndarray:
        return base.position(time_s) + anchor

    def velocity(time_s: float) -> np.ndarray:
        return base.velocity(time_s)

    def yaw(time_s: float) -> float:
        return float(base.yaw(time_s))

    return Trajectory(position=position, velocity=velocity, yaw=yaw)


def build_reference_trajectory(anchor_ned: np.ndarray, args) -> Trajectory:
    """Instantiate the trajectory to fly (edit as desired)."""

    # Choose whichever base trajectory you need and adjust parameters.

    # base = circle_trajectory(
    #     radius_m=15.0,
    #     altitude_m=-5.0,
    #     period_s=30.0,
    #     phase_rad=0.0,
    # )

    # base = line_trajectory(speed_m_s=4/.0, heading_rad=math.radians(0.0), altitude_m=0.0, heading_rad=math.degrees(args.heading_deg))

    # base = sinusoid_trajectory(
    #     forward_speed_m_s=4.0, y_amplitude_m=-10.0, y_frequency_hz=0.05
    # )

    base = zigzag_trajectory(segment_length_m=40.0, num_segments=8, speed_m_s=3, heading_deg=args.heading_deg)

    return offset_trajectory(base, anchor_ned)


async def wait_for_connection(drone: System, timeout_s: float = 30.0) -> None:
    start = time.monotonic()
    async for state in drone.core.connection_state():
        if state.is_connected:
            return
        if (time.monotonic() - start) > timeout_s:
            raise TimeoutError("Timed out waiting for vehicle connection")
        await asyncio.sleep(0.5)


async def wait_for_local_position(
    cache: TelemetryCache, timeout_s: float = 30.0, freshness_s: float = 0.5
) -> np.ndarray:
    """Return a recent local-position sample from the telemetry cache."""

    start = time.monotonic()
    while True:
        timestamp = cache.position_timestamp
        if timestamp == timestamp:  # NaN check
            age = time.monotonic() - timestamp
            if age <= freshness_s:
                return cache.position_ned.copy()
        if (time.monotonic() - start) > timeout_s:
            break
        await asyncio.sleep(0.05)

    raise TimeoutError("Timed out waiting for fresh local position data")


async def ensure_armed(drone: System) -> None:
    try:
        await drone.action.arm()
    except ActionError as exc:
        if "Armed" not in str(exc):
            raise
    async for armed in drone.telemetry.armed():
        if armed:
            return
        await asyncio.sleep(0.2)


async def start_control_mode(
    drone: System,
    initial_position: PositionNedYaw,
    initial_velocity: VelocityNedYaw,
) -> None:
    await send_position_velocity_setpoint(drone, initial_position, initial_velocity)
    try:
        await drone.offboard.start()
    except OffboardError as exc:
        raise RuntimeError(f"Failed to enter offboard/guided mode: {exc}") from exc


async def stop_control_mode(drone: System) -> None:
    try:
        await drone.offboard.stop()
    except OffboardError:
        pass


async def telemetry_position_task(drone: System, cache: TelemetryCache) -> None:
    async for sample in drone.telemetry.position_velocity_ned():
        cache.position_ned = np.array(
            [
                sample.position.north_m,
                sample.position.east_m,
                sample.position.down_m,
            ],
            dtype=float,
        )
        cache.velocity_ned = np.array(
            [
                sample.velocity.north_m_s,
                sample.velocity.east_m_s,
                sample.velocity.down_m_s,
            ],
            dtype=float,
        )
        cache.position_timestamp = time.monotonic()


async def telemetry_attitude_task(drone: System, cache: TelemetryCache) -> None:
    async for attitude in drone.telemetry.attitude_euler():
        cache.yaw_rad = math.radians(attitude.yaw_deg)
        cache.yaw_timestamp = time.monotonic()


def mavsdk_command_from_pv(
    command: np.ndarray,
    velocity: np.ndarray,
    yaw_rad: float,
) -> tuple[PositionNedYaw, VelocityNedYaw]:
    yaw_deg = math.degrees(yaw_rad)
    position = PositionNedYaw(
        float(command[0]), float(command[1]), float(command[2]), yaw_deg
    )
    velocity_cmd = VelocityNedYaw(
        float(velocity[0]), float(velocity[1]), float(velocity[2]), yaw_deg
    )
    return position, velocity_cmd


async def send_position_velocity_setpoint(
    drone: System,
    position: PositionNedYaw,
    velocity: VelocityNedYaw,
) -> None:
    await drone.offboard.set_position_ned(position)
    await drone.offboard.set_velocity_ned(velocity)


def prepare_log(log_path: Path, anchor_ned: np.ndarray) -> tuple[csv.writer, IO[str]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", newline="")
    writer = csv.writer(log_file)

    header = [
        "t_s",
        "plan_x",
        "plan_y",
        "plan_z",
        "plan_yaw_rad",
        "cmd_vx",
        "cmd_vy",
        "cmd_vz",
        "actual_x",
        "actual_y",
        "actual_z",
        "actual_vx",
        "actual_vy",
        "actual_vz",
        "actual_yaw_rad",
        "telemetry_age_s",
    ]
    writer.writerow(header)
    log_file.flush()
    return writer, log_file


async def run_path_following(
    drone: System,
    trajectory: Trajectory,
    duration_s: float,
    command_rate_hz: float,
    plan_update_period_s: float,
    planner_delay_s: float,
    follower_config: PathFollowerConfig,
    log_writer: csv.writer,
    log_stream,
    telemetry_cache: TelemetryCache,
) -> None:
    follower = PositionVelocityPathFollower(follower_config)
    plan_offsets = DEFAULT_PLAN_LOOKAHEAD_OFFSETS
    last_plan_update = -math.inf

    start_monotonic = time.monotonic()

    # Prime the follower with an initial plan and command.
    plan, plan_timestamp = build_plan(
        trajectory,
        plan_offsets,
        now_s=0.0,
        planner_delay_s=planner_delay_s,
    )
    follower.handle_plan(plan)
    try:
        initial_result = follower.next_command(0.0)
    except PathFollowerException as exc:
        raise RuntimeError(f"Failed to produce initial command: {exc}") from exc

    initial_position, initial_velocity = mavsdk_command_from_pv(
        initial_result.command.position_ned,
        initial_result.command.velocity_ned_ff,
        initial_result.command.yaw_heading
        if initial_result.command.yaw_heading is not None
        else 0.0,
    )

    await start_control_mode(drone, initial_position, initial_velocity)

    try:
        dt = 1.0 / command_rate_hz
        while True:
            now_monotonic = time.monotonic()
            elapsed = now_monotonic - start_monotonic
            if elapsed >= duration_s:
                break

            if (elapsed - last_plan_update) >= plan_update_period_s - 1e-9:
                plan, plan_timestamp = build_plan(
                    trajectory,
                    plan_offsets,
                    now_s=elapsed,
                    planner_delay_s=planner_delay_s,
                )
                follower.handle_plan(plan)
                last_plan_update = elapsed

            try:
                result = follower.next_command(elapsed)
            except PathFollowerException as exc:
                print(f"[WARN] Path follower exception at t={elapsed:.2f}s: {exc}")
                await asyncio.sleep(dt)
                continue

            cmd = result.command
            yaw_rad = cmd.yaw_heading if cmd.yaw_heading is not None else 0.0
            position_cmd, velocity_cmd = mavsdk_command_from_pv(
                cmd.position_ned,
                cmd.velocity_ned_ff,
                yaw_rad,
            )

            await send_position_velocity_setpoint(drone, position_cmd, velocity_cmd)

            actual_pos = telemetry_cache.position_ned.copy()
            actual_vel = telemetry_cache.velocity_ned.copy()
            actual_yaw = telemetry_cache.yaw_rad
            telemetry_age = (
                now_monotonic - telemetry_cache.position_timestamp
                if telemetry_cache.position_timestamp
                == telemetry_cache.position_timestamp
                else float("nan")
            )

            log_writer.writerow(
                [
                    elapsed,
                    cmd.position_ned[0],
                    cmd.position_ned[1],
                    cmd.position_ned[2],
                    yaw_rad,
                    cmd.velocity_ned_ff[0],
                    cmd.velocity_ned_ff[1],
                    cmd.velocity_ned_ff[2],
                    actual_pos[0],
                    actual_pos[1],
                    actual_pos[2],
                    actual_vel[0],
                    actual_vel[1],
                    actual_vel[2],
                    actual_yaw,
                    telemetry_age,
                ]
            )
            log_stream.flush()

            await asyncio.sleep(max(0.0, dt - (time.monotonic() - now_monotonic)))
    finally:
        await stop_control_mode(drone)


async def async_main(args: argparse.Namespace) -> None:
    drone = System()
    await drone.connect(system_address=args.connection)
    await wait_for_connection(drone)

    telemetry_rate_hz = max(1.0, args.command_rate_hz)
    await drone.telemetry.set_rate_position_velocity_ned(telemetry_rate_hz)
    await drone.telemetry.set_rate_attitude_euler(telemetry_rate_hz)

    telemetry_cache = TelemetryCache()
    tasks = [
        asyncio.create_task(telemetry_position_task(drone, telemetry_cache)),
        asyncio.create_task(telemetry_attitude_task(drone, telemetry_cache)),
    ]

    log_stream = None
    try:
        await ensure_armed(drone)

        anchor = await wait_for_local_position(telemetry_cache)
        print(
            f"[INFO] Anchor position NED: north={anchor[0]:.2f} m, east={anchor[1]:.2f} m, down={anchor[2]:.2f} m"
        )

        trajectory = build_reference_trajectory(anchor, args)

        log_dir = Path(args.log_dir)
        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"flight_path_follow_{timestamp}.csv"
        log_writer, log_stream = prepare_log(log_path, anchor)
        print(f"[INFO] Logging telemetry to {log_path}")

        follower_config = PathFollowerConfig(
            lookahead_offset_s=0.5,
            max_plan_age_s=max(
                5.0, args.plan_update_period_s + args.planner_delay_s + 1.0
            ),
        )

        await run_path_following(
            drone=drone,
            trajectory=trajectory,
            duration_s=args.duration,
            command_rate_hz=args.command_rate_hz,
            plan_update_period_s=args.plan_update_period_s,
            planner_delay_s=args.planner_delay_s,
            follower_config=follower_config,
            log_writer=log_writer,
            log_stream=log_stream,
            telemetry_cache=telemetry_cache,
        )

        print("[INFO] Path following completed.")
    finally:
        if log_stream is not None:
            try:
                log_stream.close()
            except Exception:
                pass
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fly path follower trajectories via MAVSDK."
    )
    parser.add_argument(
        "--connection",
        required=True,
        help="MAVSDK system address, e.g. udp://:14550",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="Flight duration in seconds.",
    )
    parser.add_argument(
        "--command-rate-hz",
        type=float,
        default=COMMAND_RATE_HZ_DEFAULT,
        help="Command publication rate (Hz).",
    )
    parser.add_argument(
        "--plan-update-period-s",
        type=float,
        default=PLAN_UPDATE_PERIOD_S_DEFAULT,
        help="How often to rebuild the reference plan (seconds).",
    )
    parser.add_argument(
        "--planner-delay-s",
        type=float,
        default=PLANNER_DELAY_S_DEFAULT,
        help="Assumed planner delay when stamping plan timestamps (seconds).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory to store telemetry logs.",
    )
    parser.add_argument(
        "--heading-deg",
        type=float,
        default=0.0,
        help="Heading for trajectories that take this argument (degrees)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("[INFO] Flight interrupted by user")


if __name__ == "__main__":
    main()
