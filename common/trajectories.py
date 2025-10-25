from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    """Bundle of callables describing a reference trajectory in NED coordinates."""

    position: Callable[[float], np.ndarray]
    velocity: Callable[[float], np.ndarray]
    yaw: Callable[[float], float]


def sinusoid_trajectory(
    forward_speed_m_s: float = 3.0,
    y_amplitude_m: float = 10.0,
    y_frequency_hz: float = 0.2,
    z_amplitude_m: float = 0.0,
    z_frequency_hz: float = 0.0,
) -> Trajectory:
    omega_y = 2.0 * math.pi * y_frequency_hz
    omega_z = 2.0 * math.pi * z_frequency_hz

    def position(time_s: float) -> np.ndarray:
        x = forward_speed_m_s * time_s
        y = y_amplitude_m * math.sin(omega_y * time_s)
        z = z_amplitude_m * math.sin(omega_z * time_s)
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
    altitude_m: float = 0.0,
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


def zigzag_trajectory(
    *,
    anchor_ned: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
    heading_deg: float = 0.0,
    segment_length_m: float = 40.0,
    num_segments: int = 8,
    offset_angle_deg: float = 45.0,
    speed_m_s: float = 5.0,
    start_with_positive_offset: bool = True,
) -> Trajectory:
    if num_segments <= 0:
        raise ValueError("num_segments must be positive")
    if segment_length_m <= 0.0:
        raise ValueError("segment_length_m must be positive")
    if speed_m_s <= 0.0:
        raise ValueError("speed_m_s must be positive")

    anchor = np.asarray(anchor_ned, dtype=float)
    if anchor.shape != (3,):
        raise ValueError("anchor_ned must be length-3")

    heading_rad = math.radians(heading_deg)
    offset_rad = math.radians(offset_angle_deg)
    seg_dt = segment_length_m / speed_m_s
    total_duration_s = num_segments * seg_dt

    vertices = np.empty((num_segments + 1, 3), dtype=float)
    vertices[0] = anchor

    segment_headings_rad = np.empty(num_segments, dtype=float)
    segment_unit_vectors = np.empty((num_segments, 3), dtype=float)

    sign = 1.0 if start_with_positive_offset else -1.0
    for idx in range(num_segments):
        seg_heading = heading_rad + sign * offset_rad
        dx = segment_length_m * math.cos(seg_heading)
        dy = segment_length_m * math.sin(seg_heading)
        vertices[idx + 1] = vertices[idx] + np.array([dx, dy, 0.0])
        segment_headings_rad[idx] = seg_heading

        direction = vertices[idx + 1] - vertices[idx]
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise ValueError("Generated zig-zag segment has zero length")
        segment_unit_vectors[idx] = direction / norm
        sign *= -1.0

    def _segment_index_and_alpha(time_s: float) -> tuple[int, float]:
        clamped_t = max(0.0, float(time_s))
        if clamped_t >= total_duration_s:
            return num_segments - 1, 1.0
        idx = int(clamped_t / seg_dt)
        idx = min(idx, num_segments - 1)
        seg_start_t = idx * seg_dt
        alpha = (clamped_t - seg_start_t) / seg_dt if seg_dt > 0.0 else 0.0
        return idx, alpha

    def position(time_s: float) -> np.ndarray:
        if time_s <= 0.0:
            return vertices[0].copy()
        idx, alpha = _segment_index_and_alpha(time_s)
        if alpha >= 1.0:
            return vertices[idx + 1].copy()
        return (1.0 - alpha) * vertices[idx] + alpha * vertices[idx + 1]

    def velocity(time_s: float) -> np.ndarray:
        if time_s < 0.0 or time_s >= total_duration_s:
            return np.zeros(3, dtype=float)
        idx, _ = _segment_index_and_alpha(time_s)
        return segment_unit_vectors[idx] * speed_m_s

    def yaw(time_s: float) -> float:
        if time_s < 0.0:
            return float(math.remainder(heading_rad, 2.0 * math.pi))
        idx, _ = _segment_index_and_alpha(time_s)
        if time_s >= total_duration_s:
            return float(math.remainder(segment_headings_rad[-1], 2.0 * math.pi))
        return float(math.remainder(segment_headings_rad[idx], 2.0 * math.pi))

    return Trajectory(position=position, velocity=velocity, yaw=yaw)
