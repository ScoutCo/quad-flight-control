from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from .commands import PositionVelocityCommand
from .math_utils import (
    normalize_vector,
    quat_normalize,
    quat_to_euler,
    quat_slerp,
    quat_to_rotation_matrix,
    quaternion_error,
    quaternion_error_to_rotation_vector,
    rotation_matrix_to_quat,
)

from .config import SimulatorConfig
from .states import State


@dataclass
class SimulationStep:
    time_s: float
    state: State
    commanded_accel_ned: np.ndarray
    filtered_accel_ned: np.ndarray


class Simulator:
    """First-order filtered position/velocity tracker with simple attitude dynamics."""

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        self._config = config or SimulatorConfig()
        self.dt = float(self._config.dt)
        self.state = State()
        self.time_s = 0.0
        self._accel_ned = np.zeros(3, dtype=float)
        self._target_yaw = float(self._config.initial_yaw_rad)

    def reset(self, state: State | None = None, time_s: float = 0.0) -> None:
        self.state = state.copy() if state is not None else State()
        self.time_s = float(time_s)
        self._accel_ned[:] = 0.0
        self._target_yaw = _yaw_from_quaternion(self.state.quaternion_bn)

    def step(
        self, command: PositionVelocityCommand, dt: float | None = None
    ) -> SimulationStep:
        dt = float(dt if dt is not None else self.dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        # Blend position feedforward/feedback into a bounded velocity target.
        pos_err = (
            np.asarray(command.position_ned, dtype=float) - self.state.position_ned
        )
        vel_ff = np.asarray(command.velocity_ned_ff, dtype=float)
        desired_velocity = vel_ff + self._config.position_gain * pos_err
        desired_velocity = _limit_vector_norm(
            desired_velocity, self._config.max_speed_m_s
        )

        # Close the velocity loop and constrain the commanded acceleration.
        vel_err = desired_velocity - self.state.velocity_ned
        accel_cmd = (
            np.asarray(command.accel_ned_ff, dtype=float)
            + self._config.velocity_gain * vel_err
        )
        accel_cmd = _limit_acceleration(accel_cmd, self._config)

        # First-order acceleration filter emulating actuator/inner loop lag.
        tau_acc = max(self._config.accel_time_constant, 1e-3)
        alpha_acc = dt / (tau_acc + dt)
        self._accel_ned += alpha_acc * (accel_cmd - self._accel_ned)
        self._accel_ned = _limit_acceleration(self._accel_ned, self._config)

        # Integrate translational state with the filtered acceleration.
        self.state.velocity_ned += self._accel_ned * dt
        self.state.position_ned += self.state.velocity_ned * dt

        if command.yaw_heading is not None:
            yaw_cmd = float(command.yaw_heading)
            yaw_err = math.remainder(yaw_cmd - self._target_yaw, 2.0 * math.pi)
            max_delta = max(0.0, float(self._config.max_yaw_rate_rad_s) * dt)
            if max_delta > 0.0 and abs(yaw_err) > max_delta:
                yaw_err = math.copysign(max_delta, yaw_err)
            self._target_yaw = float(
                math.remainder(self._target_yaw + yaw_err, 2.0 * math.pi)
            )

        # Compute attitude tracking target from net thrust direction + yaw.
        desired_quat = _compute_desired_attitude(
            self._accel_ned, self._target_yaw, self._config
        )

        q_prev = self.state.quaternion_bn
        tau_att = max(self._config.attitude_time_constant, 1e-3)
        alpha_att = dt / (tau_att + dt)
        q_blend = quat_slerp(q_prev, desired_quat, alpha_att)
        self.state.quaternion_bn = quat_normalize(q_blend)
        self._enforce_tilt_limit()

        q_now = self.state.quaternion_bn
        q_delta = quaternion_error(q_now, q_prev)
        rot_vec = quaternion_error_to_rotation_vector(q_delta)
        self.state.angular_velocity_body = rot_vec / max(dt, 1e-6)

        self.time_s += dt
        return SimulationStep(
            time_s=self.time_s,
            state=self.state.copy(),
            commanded_accel_ned=accel_cmd.copy(),
            filtered_accel_ned=self._accel_ned.copy(),
        )

    def run(
        self,
        final_time_s: float,
        command_fn: Callable[[float, State], PositionVelocityCommand],
        progress_callback: Callable[[SimulationStep], None] | None = None,
    ) -> list[SimulationStep]:
        steps = int(np.ceil((final_time_s - self.time_s) / self.dt))
        history: list[SimulationStep] = []
        for _ in range(max(0, steps)):
            cmd = command_fn(self.time_s, self.state.copy())
            step = self.step(cmd, dt=self.dt)
            history.append(step)
            if progress_callback is not None:
                progress_callback(step)
        return history

    def _enforce_tilt_limit(self) -> None:
        max_tilt_rad = math.radians(self._config.max_tilt_deg)
        max_tilt_cos = math.cos(max_tilt_rad)
        rot_bn = quat_to_rotation_matrix(self.state.quaternion_bn)
        body_z = rot_bn[:, 2]
        if body_z[2] >= max_tilt_cos:
            return

        yaw = math.atan2(rot_bn[1, 0], rot_bn[0, 0])
        horiz = body_z[:2]
        horiz_norm = float(np.linalg.norm(horiz))
        allowed_horiz = math.sqrt(max(0.0, 1.0 - max_tilt_cos**2))
        if horiz_norm > 1e-9 and allowed_horiz > 0.0:
            horiz = horiz * (allowed_horiz / horiz_norm)
        else:
            horiz = np.zeros(2, dtype=float)

        thrust_dir = np.array([horiz[0], horiz[1], max_tilt_cos], dtype=float)
        thrust_dir = normalize_vector(thrust_dir)

        x_ref = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=float)
        if np.linalg.norm(x_ref) < 1e-6:
            x_ref = np.array([1.0, 0.0, 0.0], dtype=float)

        y_body = np.cross(thrust_dir, x_ref)
        if np.linalg.norm(y_body) < 1e-6:
            x_ref = np.array(
                [
                    math.cos(yaw + math.pi / 2.0),
                    math.sin(yaw + math.pi / 2.0),
                    0.0,
                ],
                dtype=float,
            )
            y_body = np.cross(thrust_dir, x_ref)
        y_body = normalize_vector(y_body)
        x_body = np.cross(y_body, thrust_dir)
        x_body = normalize_vector(x_body)

        rot_bn = np.column_stack((x_body, y_body, thrust_dir))
        quat_limited = rotation_matrix_to_quat(rot_bn)

        # Choose the quaternion closest to the current state to avoid sign flips.
        if float(np.dot(quat_limited, self.state.quaternion_bn)) < 0.0:
            quat_limited = -quat_limited

        self.state.quaternion_bn = quat_limited


def _yaw_from_quaternion(q: np.ndarray) -> float:
    roll, pitch, yaw = quat_to_euler(q)
    return float(yaw)


def _compute_desired_attitude(
    accel_ned: np.ndarray,
    yaw_target: float,
    config: SimulatorConfig,
) -> np.ndarray:
    # Compute body Z axis that counteracts gravity while tracking commanded accel.
    gravity_vec = np.array([0.0, 0.0, config.gravity_m_s2], dtype=float)
    thrust_dir = gravity_vec - accel_ned
    thrust_dir = normalize_vector(thrust_dir)
    if np.linalg.norm(thrust_dir) < 1e-6:
        thrust_dir = np.array([0.0, 0.0, 1.0], dtype=float)

    # Respect the configured tilt envelope by constraining horizontal thrust.
    max_tilt_cos = math.cos(math.radians(config.max_tilt_deg))
    if thrust_dir[2] < max_tilt_cos:
        horiz = thrust_dir[:2]
        scale = math.sqrt(max(0.0, 1.0 - max_tilt_cos**2))
        horiz = _limit_vector_norm(horiz, scale)
        thrust_dir = normalize_vector(np.array([horiz[0], horiz[1], max_tilt_cos]))

    # Build orthonormal body axes that match the desired yaw heading.
    x_ref = np.array([math.cos(yaw_target), math.sin(yaw_target), 0.0], dtype=float)
    if np.linalg.norm(x_ref) < 1e-6:
        x_ref = np.array([1.0, 0.0, 0.0], dtype=float)

    y_body = np.cross(thrust_dir, x_ref)
    if np.linalg.norm(y_body) < 1e-6:
        x_ref = np.array(
            [
                math.cos(yaw_target + math.pi / 2.0),
                math.sin(yaw_target + math.pi / 2.0),
                0.0,
            ],
            dtype=float,
        )
        y_body = np.cross(thrust_dir, x_ref)
    y_body = normalize_vector(y_body)
    x_body = np.cross(y_body, thrust_dir)
    x_body = normalize_vector(x_body)

    rot_bn = np.column_stack((x_body, y_body, thrust_dir))
    return rotation_matrix_to_quat(rot_bn)


def _limit_vector_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    if max_norm <= 0.0:
        return vec
    norm = float(np.linalg.norm(vec))
    if norm <= max_norm or norm == 0.0:
        return vec
    return vec * (max_norm / norm)


def _limit_acceleration(accel: np.ndarray, config: SimulatorConfig) -> np.ndarray:
    accel = np.asarray(accel, dtype=float)
    horiz = accel[:2]
    max_horiz = config.gravity_m_s2 * math.tan(math.radians(config.max_tilt_deg))
    horiz = _limit_vector_norm(horiz, max_horiz)
    limited = accel.copy()
    limited[:2] = horiz
    limited[2] = float(
        np.clip(
            limited[2], -config.max_vertical_accel_m_s2, config.max_vertical_accel_m_s2
        )
    )
    if config.max_accel_m_s2 > 0.0:
        limited = _limit_vector_norm(limited, config.max_accel_m_s2)
    return limited
