from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    """Bundle of callables describing a reference trajectory in NED coordinates."""

    position: Callable[[float], np.ndarray]
    velocity: Callable[[float], np.ndarray]
    yaw: Callable[[float], float]


def sinusoid_trajectory(
    forward_speed_m_s: float = 3.0,
    y_amplitude_m: float = 2.0,
    y_frequency_hz: float = 0.2,
    z_base_m: float = -5.0,
    z_amplitude_m: float = 0.8,
    z_frequency_hz: float = 0.3,
) -> Trajectory:
    omega_y = 2.0 * math.pi * y_frequency_hz
    omega_z = 2.0 * math.pi * z_frequency_hz

    def position(time_s: float) -> np.ndarray:
        x = forward_speed_m_s * time_s
        y = y_amplitude_m * math.sin(omega_y * time_s)
        z = z_base_m + z_amplitude_m * math.sin(omega_z * time_s)
        return np.array([x, y, z], dtype=float)

    def velocity(time_s: float) -> np.ndarray:
        dx = forward_speed_m_s
        dy = y_amplitude_m * omega_y * math.cos(omega_y * time_s)
        dz = z_amplitude_m * omega_z * math.cos(omega_z * time_s)
        return np.array([dx, dy, dz], dtype=float)

    def yaw(time_s: float) -> float:
        vel = velocity(time_s)
        return float(math.atan2(vel[1], vel[0]))

    return Trajectory(position=position, velocity=velocity, yaw=yaw)


def circle_trajectory(
    radius_m: float = 6.0,
    altitude_m: float = -3.0,
    period_s: float = 24.0,
    phase_rad: float = 0.0,
) -> Trajectory:
    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive")
    if period_s <= 0.0:
        raise ValueError("period_s must be positive")

    omega = 2.0 * math.pi / period_s
    linear_speed = radius_m * omega

    def position(time_s: float) -> np.ndarray:
        angle = omega * time_s + phase_rad
        x = radius_m * math.cos(angle)
        y = radius_m * math.sin(angle)
        z = altitude_m
        return np.array([x, y, z], dtype=float)

    def velocity(time_s: float) -> np.ndarray:
        angle = omega * time_s + phase_rad
        vx = -linear_speed * math.sin(angle)
        vy = linear_speed * math.cos(angle)
        vz = 0.0
        return np.array([vx, vy, vz], dtype=float)

    def yaw(time_s: float) -> float:
        angle = omega * time_s + phase_rad
        yaw = angle + math.pi / 2.0
        return float(math.remainder(yaw, 2.0 * math.pi))

    return Trajectory(position=position, velocity=velocity, yaw=yaw)


def line_trajectory(
    speed_m_s: float = 4.0,
    heading_rad: float = 0.0,
    altitude_m: float = -2.5,
    lateral_offset_m: float = 0.0,
) -> Trajectory:
    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)

    def position(time_s: float) -> np.ndarray:
        distance = speed_m_s * time_s
        x = distance * cos_h - lateral_offset_m * sin_h
        y = distance * sin_h + lateral_offset_m * cos_h
        z = altitude_m
        return np.array([x, y, z], dtype=float)

    def velocity(_: float) -> np.ndarray:
        return np.array([speed_m_s * cos_h, speed_m_s * sin_h, 0.0], dtype=float)

    def yaw(_: float) -> float:
        return float(heading_rad)

    return Trajectory(position=position, velocity=velocity, yaw=yaw)
