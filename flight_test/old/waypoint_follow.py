#!/usr/bin/env python3

"""
Connect to ArduPilot Copter SITL (default UDP 127.0.0.1:14550) and send 20 Hz
GUIDED position+velocity targets to track a reference trajectory defined in the
vehicle's *body frame* (x fwd, y right, z down). Uses SET_POSITION_TARGET_LOCAL_NED.

Ref trajectory demo: level circle of radius 100 m at 5 m AGL (z = -5 NED).
"""

import math
import time
from dataclasses import dataclass
from typing import Tuple, Optional

from pymavlink import mavutil


# ----------------------------- Utilities -------------------------------------


@dataclass
class LocalNED:
    x: float
    y: float
    z: float  # NED: down is positive


def rz(
    yaw_rad: float,
) -> Tuple[
    Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]
]:
    """Rotation matrix Rz(yaw) for transforming body -> local NED (x fwd, y right, z down)."""
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def rot_apply(R, v):
    """Apply 3x3 rotation matrix R to 3-vector v."""
    return (
        R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
        R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
        R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
    )


# --------------------------- Reference path ----------------------------------


class BodyFrameReference:
    """
    Example body-frame reference:
      - Circle of radius R in the horizontal plane
      - Constant altitude z_ref (NED, down positive)
      - Angular rate omega (rad/s)
    Returns (pos_b, vel_b) in BODY frame coordinates.
    """

    def __init__(
        self, radius_m: float = 100.0, period_s: float = 20.0, z_ref_ned: float = -5.0
    ):
        self.R = radius_m
        self.omega = 2 * math.pi / period_s
        self.z_ref = z_ref_ned

    def eval(
        self, t: float
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        # Position in body frame
        x_b = self.R * math.cos(self.omega * t)
        y_b = self.R * math.sin(self.omega * t)
        z_b = self.z_ref
        # Velocity in body frame (time derivative)
        vx_b = -self.R * self.omega * math.sin(self.omega * t)
        vy_b = self.R * self.omega * math.cos(self.omega * t)
        vz_b = 0.0
        return (x_b, y_b, z_b), (vx_b, vy_b, vz_b)


class LocalNEDSinuousReference:
    """
    Local NED (x=north, y=east, z=down) reference that progresses generally north
    while oscillating sinusoidally in Y (east) and Z (vertical) with different
    frequencies. Yaw is aligned with the velocity tangent; psi is yaw rate.

    eval(t) -> ((x, y, z, yaw), (vx, vy, vz, psi))
      - Positions in meters (NED), yaw in radians.
      - Velocities in m/s, psi in rad/s.

    Params:
      v_north_mps   : mean forward speed along +X (north), > 0 keeps tangent well-defined
      y_amp_m       : amplitude of lateral (east) oscillation
      y_freq_hz     : frequency (Hz) of lateral oscillation
      z_mean_m      : mean altitude in NED (down positive; e.g., -5 = 5 m AGL)
      z_amp_m       : amplitude of vertical oscillation (NED)
      z_freq_hz     : frequency (Hz) of vertical oscillation
      x0, y0        : initial NED offsets (meters)
      yaw_offset_rad: constant yaw offset added to tangent (optional)
    """

    def __init__(
        self,
        v_north_mps: float = 5.0,
        y_amp_m: float = 20.0,
        y_freq_hz: float = 0.05,  # 20 s lateral period
        z_mean_m: float = -30.0,  # ~30 m AGL in NED
        z_amp_m: float = 1.0,
        z_freq_hz: float = 0.1,  # 10 s vertical period
        x0: float = 0.0,
        y0: float = 0.0,
        yaw_offset_rad: float = 0.0,
    ):
        import math

        self.math = math
        self.vx = float(v_north_mps)
        self.Ay = float(y_amp_m)
        self.oy = 2.0 * math.pi * float(y_freq_hz)

        self.z_mean = float(z_mean_m)
        self.Az = float(z_amp_m)
        self.oz = 2.0 * math.pi * float(z_freq_hz)

        self.x0 = float(x0)
        self.y0 = float(y0)
        self.yaw_off = float(yaw_offset_rad)

        self._eps = 1e-6  # for safe division in yaw-rate

    def eval(self, t: float):
        """Return ((x, y, z, yaw), (vx, vy, vz, psi)) at time t (seconds)."""
        m = self.math

        # Position (NED)
        x = self.x0 + self.vx * t
        y = self.y0 + self.Ay * m.sin(self.oy * t)
        z = self.z_mean + self.Az * m.sin(self.oz * t)

        # Velocity (NED)
        vx = self.vx
        vy = self.Ay * self.oy * m.cos(self.oy * t)
        vz = self.Az * self.oz * m.cos(self.oz * t)

        # Tangent-aligned yaw
        yaw = m.atan2(vy, vx) + self.yaw_off

        # Yaw rate psi = d/dt atan2(vy, vx) = (vx*ay - vy*ax)/(vx^2 + vy^2)
        ax = 0.0
        ay = -self.Ay * (self.oy**2) * m.sin(self.oy * t)
        denom = max(self._eps, (vx * vx + vy * vy))
        psi = (vx * ay - vy * ax) / denom

        return (x, y, z, yaw), (vx, vy, vz, psi)


# ----------------------------- MAV helpers -----------------------------------


def wait_heartbeat(master: mavutil.mavfile, timeout: float = 10.0):
    msg = master.wait_heartbeat(timeout=timeout)
    if msg is None:
        raise TimeoutError("No HEARTBEAT within timeout.")


def set_mode_guided(master: mavutil.mavfile):
    mapping = master.mode_mapping()
    if "GUIDED" not in mapping:
        raise RuntimeError("GUIDED mode not available on this vehicle.")
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mapping["GUIDED"],
    )


def arm(master: mavutil.mavfile, timeout: float = 15.0):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    t0 = time.time()
    while not master.motors_armed():
        master.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
        if time.time() - t0 > timeout:
            raise TimeoutError("Arming timeout.")


def disarm(master: mavutil.mavfile, timeout: float = 10.0):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    t0 = time.time()
    while master.motors_armed():
        master.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
        if time.time() - t0 > timeout:
            print("Warning: disarm timeout; vehicle may still be armed.")
            return


def get_local_position(
    master: mavutil.mavfile, wait: bool = True
) -> Optional[LocalNED]:
    """
    LOCAL_POSITION_NED.x/y/z are in meters in NED frame.
    z is positive down. Returns None if not available and wait=False.
    """
    msg = master.recv_match(type="LOCAL_POSITION_NED", blocking=wait, timeout=1.0)
    if not msg:
        return None
    return LocalNED(float(msg.x), float(msg.y), float(msg.z))


def get_yaw(master: mavutil.mavfile, wait: bool = True) -> Optional[float]:
    """
    Returns yaw (rad) from ATTITUDE message (NED yaw, range -pi..pi).
    """
    msg = master.recv_match(type="ATTITUDE", blocking=wait, timeout=1.0)
    if not msg:
        return None
    return float(msg.yaw)


def send_pos_vel_target_local_ned(
    master: mavutil.mavfile,
    pos_ned: Tuple[float, float, float],
    vel_ned: Tuple[float, float, float],
):
    """
    Send SET_POSITION_TARGET_LOCAL_NED with position + velocity control.
    We ignore accel, yaw, yaw_rate. Type mask bits 6..10 set = 1984.
    """
    type_mask = (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9) | (1 << 10)  # 1984
    master.mav.set_position_target_local_ned_send(
        int(time.time() * 1000) & 0xFFFFFFFF,  # time_boot_ms (OK to approximate)
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # coordinates in local NED
        type_mask,
        pos_ned[0],
        pos_ned[1],
        pos_ned[2],  # x, y, z (m)
        0,  # vel_ned[0],
        0,  # vel_ned[1],
        0,  # vel_ned[2],  # vx, vy, vz (m/s)
        0,
        0,
        0,  # ax, ay, az (ignored)
        0,
        0,  # yaw, yaw_rate (ignored)
    )


def send_pos_vel_yawrate_local_ned(master, pos_ned, vel_ned, yaw_rate_rad_s):
    # ignore accel + yaw (but NOT yaw_rate)
    type_mask = (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9)  # 64+128+256+512 = 960
    master.mav.set_position_target_local_ned_send(
        int(time.time() * 1000) & 0xFFFFFFFF,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        pos_ned[0],
        pos_ned[1],
        pos_ned[2],
        vel_ned[0],
        vel_ned[1],
        vel_ned[2],
        0,
        0,
        0,
        0,
        yaw_rate_rad_s,
    )


def send_pos_vel_yaw_yawrate_local_ned(
    master,
    pos_ned,  # (x, y, z)   in meters, LOCAL_NED
    vel_ned,  # (vx, vy, vz) in m/s,   LOCAL_NED
    yaw_rad,  # yaw setpoint (rad, NED frame, -pi..pi)
    yaw_rate_rad_s,  # yaw rate (rad/s)
):
    """
    SET_POSITION_TARGET_LOCAL_NED with position + velocity + yaw + yaw_rate.
    - Uses LOCAL_NED. If you prefer relative control, switch the frame to BODY_OFFSET_NED.
    - Accel is ignored; 'is force' bit is cleared.
    """
    from pymavlink import mavutil
    import time

    # Ignore only accelerations (bits 6,7,8). Keep position, velocity, yaw, yaw_rate active.
    # DO NOT set bit 9 (force setpoint).
    type_mask = (1 << 6) | (1 << 7) | (1 << 8)  # = 448

    master.mav.set_position_target_local_ned_send(
        int(time.time() * 1000) & 0xFFFFFFFF,  # time_boot_ms (approx OK)
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # switch to BODY_OFFSET_NED if desired
        type_mask,
        float(pos_ned[0]),
        float(pos_ned[1]),
        float(pos_ned[2]),
        float(vel_ned[0]),
        float(vel_ned[1]),
        float(vel_ned[2]),
        0.0,
        0.0,
        0.0,  # ax, ay, az (ignored)
        float(yaw_rad),
        float(yaw_rate_rad_s),  # yaw, yaw_rate
    )


def send_pos_vel_target_body_offset(master, pos_body_offset, vel_body):
    # Use position+velocity, ignore accel+yaw+yaw_rate (same mask = 1984)
    type_mask = (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9) | (1 << 10)
    master.mav.set_position_target_local_ned_send(
        int(time.time() * 1000) & 0xFFFFFFFF,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # <-- key change
        type_mask,
        pos_body_offset[0],
        pos_body_offset[1],
        pos_body_offset[2],
        vel_body[0],
        vel_body[1],
        vel_body[2],
        0,
        0,
        0,
        0,
        0,
    )


def takeoff(master: mavutil.mavfile, altitude_m: float = 5.0, timeout: float = 30.0):
    """Command takeoff to a specified altitude (m)."""
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        altitude_m,
    )
    print(f"Taking off to {altitude_m} m AGL...")

    t0 = time.time()
    while True:
        msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1.0)
        if msg:
            alt = msg.relative_alt / 1000.0  # mm → m
            if alt >= altitude_m * 0.95:
                print(f"Reached takeoff altitude: {alt:.1f} m")
                break
        if time.time() - t0 > timeout:
            print("Takeoff timeout!")
            break


def land(master: mavutil.mavfile, timeout: float = 45.0):
    """Command LAND mode and wait until vehicle touches down and disarms."""
    mapping = master.mode_mapping()
    if "LAND" not in mapping:
        raise RuntimeError("LAND mode not available.")

    print("Switching to LAND mode...")
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mapping["LAND"],
    )

    t0 = time.time()
    landed = False
    while time.time() - t0 < timeout:
        msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1.0)
        if msg:
            alt = msg.relative_alt / 1000.0
            if alt <= 0.2:
                landed = True
                print(f"Landed at {alt:.2f} m AGL.")
                break
    if not landed:
        print("Warning: LAND timeout, forcing disarm.")
    else:
        print("Touchdown detected, waiting for auto-disarm...")

    # Wait for disarm confirmation
    t1 = time.time()
    while master.motors_armed() and time.time() - t1 < 20:
        master.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
    if not master.motors_armed():
        print("Vehicle disarmed.")
    else:
        print("Auto-disarm timeout; disarming manually.")
        disarm(master)


# ------------------------------ Main logic -----------------------------------


def main():
    # --- Connect to SITL (default UDP endpoint from ArduPilot SITL) ---
    print("Connecting to SITL at udp:127.0.0.1:14550 ...")
    master = mavutil.mavlink_connection("udp:127.0.0.1:14550", autoreconnect=True)
    wait_heartbeat(master)
    print(
        f"Heartbeat from system {master.target_system} component {master.target_component}"
    )

    # # --- GUIDED mode + arm ---
    print("Switching to GUIDED ...")
    set_mode_guided(master)

    print("Arming ...")
    arm(master)
    print("Armed.")

    print("Taking off ...")
    takeoff(master, altitude_m=5.0)

    # --- Wait for valid local position & attitude; capture initial frame ---
    print("Waiting for LOCAL_POSITION_NED and ATTITUDE ...")
    pos0 = None
    yaw0 = None
    while pos0 is None:
        pos0 = get_local_position(master, wait=True)
    while yaw0 is None:
        yaw0 = get_yaw(master, wait=True)

    # Use the *initial* body frame as the reference anchor
    # (trajectory is defined relative to the vehicle pose at t0)
    rz(yaw0)
    x0, y0, z0 = pos0.x, pos0.y, pos0.z
    print(
        f"Initial NED position: ({x0:.2f}, {y0:.2f}, {z0:.2f}) m, yaw0 = {math.degrees(yaw0):.1f}°"
    )

    # --- Build reference trajectory---

    # Body frame version
    # ref = BodyFrameReference(radius_m=50.0, period_s=30.0, z_ref_ned=-25.0)

    ref = LocalNEDSinuousReference(
        v_north_mps=5.0,
        y_amp_m=15.0,
        y_freq_hz=0.05,  # 20 s lateral period
        z_mean_m=-30.0,
        z_amp_m=4.0,
        z_freq_hz=0.1,  # 10 s vertical period
    )

    # --- Control loop at 20 Hz ---
    rate_hz = 20.0
    dt = 1.0 / rate_hz
    run_time_s = 300.0  # demo duration

    print("Streaming position+velocity targets at 20 Hz ... (Ctrl+C to stop)")
    t_start = time.time()

    try:
        while True:
            t = time.time() - t_start
            if t > run_time_s:
                break

            # ------- LOCAL FRAME SINUSIOD -----------
            (pos, deriv) = ref.eval(t)
            x, y, z, yaw = pos
            vx, vy, vz, psi = deriv

            send_pos_vel_yaw_yawrate_local_ned(master, (x, y, z), (vx, vy, vz), yaw, psi)

            # ------- LOCAL-FRAME REF VERSION ---------
            # # Reference in BODY frame
            # (x_b, y_b, z_b), (vx_b, vy_b, vz_b) = ref.eval(t)

            # # Transform to local NED using initial yaw anchor
            # vx_n, vy_n, vz_n = rot_apply(R_b_to_ned, (vx_b, vy_b, vz_b))
            # x_n, y_n, z_n = rot_apply(R_b_to_ned, (x_b, y_b, z_b))

            # # Absolute setpoint in local NED (offset from initial position)
            # pos_sp = (x0 + x_n, y0 + y_n, z0 + z_n)
            # vel_sp = (vx_n, vy_n, vz_n)
            # print(f"pos: {pos_sp}, vel: {vel_sp}")

            # # Send the setpoint
            # send_pos_vel_target_local_ned(master, pos_sp, vel_sp)
            # ----------------------------------------

            # # ------- BODY-FRAME REF VERSION ---------
            # (pos_b, vel_b) = ref.eval(t)
            # send_pos_vel_target_body_offset(master, pos_b, vel_b)
            # # ----------------------------------------

            # Keep link alive / read messages to maintain freshness
            master.recv_match(blocking=False)

            # Sleep to maintain 20 Hz
            next_tick = t_start + (math.floor((t * rate_hz) + 1) / rate_hz)
            sleep_time = max(0.0, next_tick - time.time())
            time.sleep(sleep_time if sleep_time < dt else dt)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    # --- Clean up: switch to LOITER ---
    try:
        mapping = master.mode_mapping()
        if "LOITER" in mapping:
            master.mav.set_mode_send(
                master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mapping["LOITER"],
            )
            time.sleep(1.0)
    except Exception:
        pass

    # --- Land and disarm safely ---
    print("Landing ...")
    land(master)
    print("Mission complete.")


if __name__ == "__main__":
    main()
