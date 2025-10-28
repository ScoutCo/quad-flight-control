from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SimulatorConfig:
    """Configuration parameters for the quadrotor simulator."""

    dt: float = 0.02
    position_gain: float = 0.8
    velocity_gain: float = 2.8
    accel_time_constant: float = 0.15
    attitude_time_constant: float = 0.3
    attitude_rate_time_constant: float = 0.18
    max_accel_m_s2: float = 9.0
    max_vertical_accel_m_s2: float = 6.0
    max_speed_m_s: float = 12.0
    gravity_m_s2: float = 9.80665
    max_tilt_deg: float = 35.0
    initial_yaw_rad: float = 0.0
    max_yaw_rate_rad_s: float = math.radians(180.0)
