from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .commands import PositionVelocityCommand
from .math_utils import (
    normalize_vector,
    rotation_matrix_to_quat,
    quaternion_error,
    quaternion_error_to_rotation_vector,
    quat_to_euler,
    quat_normalize,
)
from .parameters import ControllerGains, VehicleParameters
from .states import SixDofState


class VectorPID:
    def __init__(self, kp: np.ndarray, ki: np.ndarray, kd: np.ndarray, integrator_limit: np.ndarray):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integrator_limit = integrator_limit
        self.integrator = np.zeros(3)
        self.prev_error = np.zeros(3)

    def reset(self) -> None:
        self.integrator[:] = 0.0
        self.prev_error[:] = 0.0

    def step(self, error: np.ndarray, dt: float) -> np.ndarray:
        self.integrator += error * dt
        self.integrator = np.clip(self.integrator, -self.integrator_limit, self.integrator_limit)
        derivative = (error - self.prev_error) / max(dt, 1e-6)
        self.prev_error = error
        return self.kp * error + self.ki * self.integrator + self.kd * derivative


class FirstOrderLag:
    def __init__(self, time_constant: float, initial: np.ndarray | float):
        self.tau = max(time_constant, 1e-6)
        self.state = np.array(initial, copy=True) if np.ndim(initial) else float(initial)

    def reset(self, value: np.ndarray | float) -> None:
        self.state = np.array(value, copy=True) if np.ndim(value) else float(value)

    def update(self, target: np.ndarray | float, dt: float) -> np.ndarray | float:
        alpha = dt / (self.tau + dt)
        self.state = (1.0 - alpha) * self.state + alpha * target
        return self.state


@dataclass
class ControllerState:
    pos_velocity_target: np.ndarray
    velocity_target: np.ndarray
    attitude_target_quat: np.ndarray
    thrust_command: float
    body_rate_command: np.ndarray
    moment_command: np.ndarray


class CascadedController:
    def __init__(self, vehicle: VehicleParameters, gains: ControllerGains):
        self.vehicle = vehicle
        self.gains = gains
        self.position_pid = VectorPID(gains.position.kp, gains.position.ki, gains.position.kd, gains.position.integrator_limit)
        self.velocity_pid = VectorPID(gains.velocity.kp, gains.velocity.ki, gains.velocity.kd, gains.velocity.integrator_limit)
        self.rate_pid = VectorPID(gains.rate.kp, gains.rate.ki, gains.rate.kd, gains.rate.integrator_limit)
        self.pos_lag = FirstOrderLag(gains.position_lag_time_constant, np.zeros(3))
        self.vel_lag = FirstOrderLag(gains.velocity_lag_time_constant, np.zeros(3))
        self.attitude_lag = FirstOrderLag(gains.attitude_lag_time_constant, np.array([1.0, 0.0, 0.0, 0.0]))
        self.rate_lag = FirstOrderLag(gains.rate_lag_time_constant, np.zeros(3))

    def reset(self) -> None:
        self.position_pid.reset()
        self.velocity_pid.reset()
        self.rate_pid.reset()
        self.pos_lag.reset(np.zeros(3))
        self.vel_lag.reset(np.zeros(3))
        self.attitude_lag.reset(np.array([1.0, 0.0, 0.0, 0.0]))
        self.rate_lag.reset(np.zeros(3))

    def update(self, state: SixDofState, command: PositionVelocityCommand, env_gravity: np.ndarray, dt: float) -> ControllerState:
        pos_error = command.position_ned - state.position_ned
        pos_correction = self.position_pid.step(pos_error, dt)
        vel_target = command.velocity_ned_ff + pos_correction
        vel_target = self.pos_lag.update(vel_target, dt).copy()

        vel_error = vel_target - state.velocity_ned
        accel_cmd = self.velocity_pid.step(vel_error, dt)
        accel_cmd = self.vel_lag.update(accel_cmd, dt).copy()

        thrust_vector_ned = self.vehicle.mass * (accel_cmd - env_gravity)
        thrust_mag = np.linalg.norm(thrust_vector_ned)
        thrust_mag = float(np.clip(thrust_mag, 0.0, np.inf))

        if thrust_mag < 1e-5:
            thrust_mag = 1e-5
            thrust_vector_ned = np.array([0.0, 0.0, -thrust_mag])

        z_body_ned = -thrust_vector_ned / thrust_mag
        yaw_heading = command.yaw_heading
        if yaw_heading is None:
            yaw_heading = quat_to_euler(state.quaternion_bn)[2]
        x_c = np.array([np.cos(yaw_heading), np.sin(yaw_heading), 0.0])
        x_body_ned = normalize_vector(x_c - np.dot(x_c, z_body_ned) * z_body_ned)
        if np.linalg.norm(x_body_ned) < 1e-6:
            x_body_ned = normalize_vector(np.array([z_body_ned[2], 0.0, -z_body_ned[0]]))
        y_body_ned = np.cross(z_body_ned, x_body_ned)
        rotation = np.column_stack((x_body_ned, y_body_ned, z_body_ned))
        attitude_target = rotation_matrix_to_quat(rotation)
        attitude_target = quat_normalize(self.attitude_lag.update(attitude_target, dt))

        attitude_error = quaternion_error(attitude_target, state.quaternion_bn)
        rotation_vec = quaternion_error_to_rotation_vector(attitude_error)
        body_rate_cmd = self.gains.attitude.kp * rotation_vec
        body_rate_cmd = self.rate_lag.update(body_rate_cmd, dt).copy()

        rate_error = body_rate_cmd - state.angular_velocity_body
        moment_cmd = self.rate_pid.step(rate_error, dt).copy()

        return ControllerState(
            pos_velocity_target=vel_target.copy(),
            velocity_target=vel_target.copy(),
            attitude_target_quat=attitude_target.copy(),
            thrust_command=thrust_mag,
            body_rate_command=body_rate_cmd.copy(),
            moment_command=moment_cmd.copy(),
        )
