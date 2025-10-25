#!/usr/bin/env python3
"""Simple MAVSDK offboard example that flies a circle in local NED coordinates.

Assumes the multirotor is already airborne and loitering before the script runs.
"""

import argparse
import asyncio
import math
from typing import Optional

from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw, VelocityNedYaw

try:
    # Available in MAVSDK >= 1.5.0
    from mavsdk.offboard import PositionVelocityNedYaw  # type: ignore
except ImportError:  # pragma: no cover - fallback for older mavsdk builds
    PositionVelocityNedYaw = None


async def wait_for_connection(drone: System, timeout: float = 30.0) -> None:
    """Wait for a vehicle connection or raise if it times out."""

    async def _monitor() -> None:
        async for state in drone.core.connection_state():
            if state.is_connected:
                return

    await asyncio.wait_for(_monitor(), timeout=timeout)


async def get_initial_position(drone: System) -> PositionNedYaw:
    """Grab the current NED position to use as the circle center."""
    async for pos_vel in drone.telemetry.position_velocity_ned():
        position = pos_vel.position
        return PositionNedYaw(
            position.north_m,
            position.east_m,
            position.down_m,
            0.0,
        )
    raise RuntimeError("Failed to sample initial NED position")


async def fly_circle(
    drone: System,
    radius: float,
    period: float,
    duration: float,
    update_rate_hz: float,
) -> None:
    """Command a circle using offboard position/velocity control."""
    circle_center = await get_initial_position(drone)

    # Provide an initial setpoint before switching to offboard mode.
    initial_yaw = circle_center.yaw_deg
    if PositionVelocityNedYaw is not None:
        await drone.offboard.set_position_velocity_ned(
            PositionVelocityNedYaw(
                circle_center,
                VelocityNedYaw(0.0, 0.0, 0.0, 0.0),
            )
        )
    else:
        await drone.offboard.set_position_ned(circle_center)
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))

    try:
        await drone.offboard.start()
    except OffboardError as error:
        raise RuntimeError(f"Failed to start offboard mode: {error}") from error

    loop = asyncio.get_running_loop()
    omega = 2.0 * math.pi / period
    update_period = 1.0 / update_rate_hz
    start_time = loop.time()
    center_north = circle_center.north_m
    center_east = circle_center.east_m
    down = circle_center.down_m

    try:
        while loop.time() - start_time < duration:
            t = loop.time() - start_time
            angle = omega * t
            north = center_north + radius * math.cos(angle)
            east = center_east + radius * math.sin(angle)
            vx = -radius * omega * math.sin(angle)
            vy = radius * omega * math.cos(angle)

            position_setpoint = PositionNedYaw(north, east, down, initial_yaw)
            velocity_setpoint = VelocityNedYaw(vx, vy, 0.0, 0.0)

            if PositionVelocityNedYaw is not None:
                await drone.offboard.set_position_velocity_ned(
                    PositionVelocityNedYaw(position_setpoint, velocity_setpoint)
                )
            else:
                await drone.offboard.set_position_ned(position_setpoint)
                await drone.offboard.set_velocity_ned(velocity_setpoint)

            await asyncio.sleep(update_period)
    finally:
        # Stop sending offboard setpoints but leave the vehicle holding its last position.
        try:
            await drone.offboard.stop()
        except OffboardError:
            pass


async def main(args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system-address",
        default="udp://:14550",
        help="Connection URI for the target vehicle (e.g. udpin://:14550 for ArduPilot)",
    )
    parser.add_argument(
        "--mavsdk-server-address",
        default=None,
        help=(
            "gRPC endpoint for an already running mavsdk_server. Leave unset to let "
            "the Python binding start an embedded server."
        ),
    )
    parser.add_argument("--radius", type=float, default=20.0, help="Circle radius in meters")
    parser.add_argument(
        "--period",
        type=float,
        default=60.0,
        help="Time in seconds to complete one circle",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="Total mission time in seconds",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=20.0,
        help="Setpoint update rate in Hz",
    )
    parsed_args = parser.parse_args(args=args)

    drone = System(mavsdk_server_address=parsed_args.mavsdk_server_address)
    print(f"Connecting to {parsed_args.system_address}...")
    await drone.connect(system_address=parsed_args.system_address)

    try:
        await wait_for_connection(drone)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            "Timed out waiting for a vehicle. Verify the UDP endpoint and that mavsdk_server is reachable."
        ) from exc

    print("Vehicle connected. Switching to offboard control...")

    await fly_circle(
        drone,
        radius=parsed_args.radius,
        period=parsed_args.period,
        duration=parsed_args.duration,
        update_rate_hz=parsed_args.rate,
    )


if __name__ == "__main__":
    asyncio.run(main())
