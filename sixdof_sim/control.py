from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .commands import PositionVelocityCommand
from .math_utils import (
    normalize_vector,
    rotation_matrix_to_quat,
    quaternion_error,
    quaternion_error_to_rotation_vector,
    quat_normalize,
    quat_to_euler,
    quat_to_rotation_matrix,
)
from .parameters import ActuatorParameters, ControllerGains, VehicleParameters
from .states import SixDofState


class VectorPID:
    def __init__(
        self,
        kp: np.ndarray,
        ki: np.ndarray,
        kd: np.ndarray,
        integrator_limit: np.ndarray,
    ):
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
        self.integrator = np.clip(
            self.integrator, -self.integrator_limit, self.integrator_limit
        )
        derivative = (error - self.prev_error) / max(dt, 1e-6)
        self.prev_error = error
        return self.kp * error + self.ki * self.integrator + self.kd * derivative

    def back_calculate(self, correction: np.ndarray, gain: np.ndarray, dt: float) -> None:
        gain = np.asarray(gain, dtype=float)
        if not np.any(gain):
            return
        self.integrator += gain * correction * dt
        self.integrator = np.clip(
            self.integrator, -self.integrator_limit, self.integrator_limit
        )


@dataclass
class ControllerState:
    pos_velocity_target: np.ndarray
    velocity_target: np.ndarray
    attitude_target_quat: np.ndarray
    thrust_command: float
    body_rate_command: np.ndarray
    moment_command: np.ndarray


class CascadedController:
    def __init__(
        self,
        vehicle: VehicleParameters,
        gains: ControllerGains,
        actuators: ActuatorParameters | None = None,
    ):
        self.vehicle = vehicle
        self.gains = gains
        self.actuator_params = actuators

        self.position_pid = VectorPID(
            gains.position.kp,
            gains.position.ki,
            gains.position.kd,
            gains.position.integrator_limit,
        )
        self.velocity_pid = VectorPID(
            gains.velocity.kp,
            gains.velocity.ki,
            gains.velocity.kd,
            gains.velocity.integrator_limit,
        )
        self.rate_pid = VectorPID(
            gains.rate.kp, gains.rate.ki, gains.rate.kd, gains.rate.integrator_limit
        )
        self.last_yaw_cmd = None

    def reset(self) -> None:
        self.position_pid.reset()
        self.velocity_pid.reset()
        self.rate_pid.reset()
        self.last_yaw_cmd = None

    def update(
        self,
        state: SixDofState,
        command: PositionVelocityCommand,
        gravity: np.ndarray,
        dt: float,
    ) -> ControllerState:
        rotation_bn = quat_to_rotation_matrix(state.quaternion_bn)
        rotation_nb = rotation_bn.T

        # Position loop in body frame
        pos_error_body = rotation_nb @ (command.position_ned - state.position_ned)
        vel_offset_body = self.position_pid.step(pos_error_body, dt)
        vel_ff_body = rotation_nb @ command.velocity_ned_ff
        vel_target_body_raw = vel_ff_body + vel_offset_body
        if self.actuator_params is not None and "max_velocity_target_xy" in self.gains.__dict__:
            vel_target_body_raw[2] = vel_ff_body[2] + vel_offset_body[2]
        vel_target_body = self._limit_vector(
            vel_target_body_raw,
            self.gains.max_velocity_target_xy,
            self.gains.max_velocity_target_z,
        )
        self.position_pid.back_calculate(
            vel_target_body - vel_target_body_raw,
            self.gains.position_anti_windup_gain,
            dt,
        )
        vel_target = rotation_bn @ vel_target_body

        # Velocity loop in body frame
        vel_body = rotation_nb @ state.velocity_ned
        vel_error_body = vel_target_body - vel_body
        accel_offset_body = self.velocity_pid.step(vel_error_body, dt)
        accel_ff_body = rotation_nb @ command.accel_ned_ff
        accel_target_body_raw = accel_ff_body + accel_offset_body
        accel_target_body_raw[2] = accel_ff_body[2] + accel_offset_body[2]
        accel_target_body = self._limit_vector(
            accel_target_body_raw,
            self.gains.max_accel_target_xy,
            self.gains.max_accel_target_z,
        )
        self.velocity_pid.back_calculate(
            accel_target_body - accel_target_body_raw,
            self.gains.velocity_anti_windup_gain,
            dt,
        )
        accel_target = rotation_bn @ accel_target_body

        # Convert desired acceleration to thrust vector
        thrust_desired = self.vehicle.mass * (accel_target - gravity)
        thrust_limited = self._limit_thrust(thrust_desired)
        thrust_mag = float(np.linalg.norm(thrust_limited))

        if thrust_mag < 1e-5:
            thrust_mag = 1e-5
            thrust_limited = np.array([0.0, 0.0, -thrust_mag])

        accel_actual = gravity + thrust_limited / self.vehicle.mass
        accel_actual_body = rotation_nb @ accel_actual
        self.velocity_pid.back_calculate(
            accel_actual_body - accel_target_body,
            self.gains.velocity_anti_windup_gain,
            dt,
        )

        # Build attitude setpoint from thrust vector
        z_body_ned = -thrust_limited / thrust_mag
        yaw_heading = command.yaw_heading
        if yaw_heading is None:
            yaw_heading = quat_to_euler(state.quaternion_bn)[2]
        x_c = np.array([np.cos(yaw_heading), np.sin(yaw_heading), 0.0])
        x_body_ned = normalize_vector(x_c - np.dot(x_c, z_body_ned) * z_body_ned)
        if np.linalg.norm(x_body_ned) < 1e-6:
            x_body_ned = normalize_vector(np.array([z_body_ned[2], 0.0, -z_body_ned[0]]))
        y_body_ned = np.cross(z_body_ned, x_body_ned)
        rotation = np.column_stack((x_body_ned, y_body_ned, z_body_ned))
        attitude_target = quat_normalize(rotation_matrix_to_quat(rotation))
        if attitude_target[0] < 0.0:
            attitude_target *= -1.0

        # Rate loop
        attitude_error = quaternion_error(attitude_target, state.quaternion_bn)
        rotation_vec = quaternion_error_to_rotation_vector(attitude_error)
        yaw_rate_ff = 0.0
        if command.yaw_heading is not None and self.last_yaw_cmd is not None:
            yaw_rate_ff = self._wrap_angle(command.yaw_heading - self.last_yaw_cmd) / max(dt, 1e-6)
        if command.yaw_heading is not None:
            self.last_yaw_cmd = command.yaw_heading
        else:
            self.last_yaw_cmd = quat_to_euler(state.quaternion_bn)[2]

        body_rate_cmd = self.gains.attitude.kp * rotation_vec
        body_rate_cmd[2] += yaw_rate_ff

        max_body_rate = getattr(self.gains, "max_body_rate", None)
        if max_body_rate is not None:
            body_rate_cmd = np.clip(body_rate_cmd, -max_body_rate, max_body_rate)

        rate_error = body_rate_cmd - state.angular_velocity_body
        moment_cmd_raw = self.rate_pid.step(rate_error, dt)

        if self.actuator_params is not None:
            max_moment = self.actuator_params.max_moment
            moment_cmd = np.clip(moment_cmd_raw, -max_moment, max_moment)
            self.rate_pid.back_calculate(
                moment_cmd - moment_cmd_raw,
                self.gains.rate_anti_windup_gain,
                dt,
            )
        else:
            moment_cmd = moment_cmd_raw

        return ControllerState(
            pos_velocity_target=vel_target.copy(),
            velocity_target=accel_target.copy(),
            attitude_target_quat=attitude_target.copy(),
            thrust_command=thrust_mag,
            body_rate_command=body_rate_cmd.copy(),
            moment_command=moment_cmd.copy(),
        )

    def _limit_vector(
        self, vector: np.ndarray, max_xy: float | None, max_z: float | None
    ) -> np.ndarray:
        limited = np.array(vector, copy=True, dtype=float)
        if max_xy is not None and max_xy > 0.0:
            horiz = limited[:2]
            norm = np.linalg.norm(horiz)
            if norm > max_xy:
                limited[:2] = horiz * (max_xy / norm)
        if max_z is not None and max_z > 0.0:
            limited[2] = float(np.clip(limited[2], -max_z, max_z))
        return limited

    def _limit_thrust(self, thrust: np.ndarray) -> np.ndarray:
        limited = np.array(thrust, copy=True, dtype=float)
        if limited[2] >= 0.0:
            if abs(limited[2]) < 1e-6:
                limited[2] = -1e-6
            else:
                limited[2] = -abs(limited[2])

        max_angle = getattr(self.gains, "max_tilt_angle_rad", None)
        if max_angle is not None and max_angle > 0.0:
            vertical = limited[2]
            horiz = limited[:2]
            horiz_norm = np.linalg.norm(horiz)
            max_horiz = abs(vertical) * np.tan(max_angle)
            if horiz_norm > max_horiz > 0.0:
                limited[:2] = horiz * (max_horiz / horiz_norm)

        if self.actuator_params is not None:
            max_thrust = float(self.actuator_params.max_thrust)
            norm = np.linalg.norm(limited)
            if norm > max_thrust > 0.0:
                limited *= max_thrust / norm

        return limited

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi
