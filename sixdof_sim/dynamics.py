from __future__ import annotations

import numpy as np

from .math_utils import quat_to_rotation_matrix
from .parameters import VehicleParameters
from .states import SixDofState


class RigidBodyDynamics:
    def __init__(self, params: VehicleParameters):
        self.params = params
        self.inv_inertia = self.params.inverse_inertia()

    def derivatives(
        self,
        state: SixDofState,
        thrust: float,
        moment_body: np.ndarray,
        gravity: np.ndarray,
        wind_ned: np.ndarray,
        external_force_body: np.ndarray,
        external_moment_body: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Express state in body frame to build forces
        rotation = quat_to_rotation_matrix(state.quaternion_bn)
        velocity_rel_ned = state.velocity_ned - wind_ned
        velocity_rel_body = rotation.T @ velocity_rel_ned

        drag_body = -self.params.drag_coeff_linear * velocity_rel_body
        thrust_body = np.array([0.0, 0.0, -thrust])

        # Net translational dynamics in both frames.
        total_force_body = thrust_body + drag_body + external_force_body
        total_force_ned = rotation @ total_force_body
        acceleration_ned = total_force_ned / self.params.mass + gravity

        omega = state.angular_velocity_body
        angular_damping = -self.params.angular_damping * omega

        # Rotational dynamics: damping and gyroscopic coupling
        total_moment_body = moment_body + angular_damping + external_moment_body
        omega_dot = self.inv_inertia @ (
            total_moment_body - np.cross(omega, self.params.inertia @ omega)
        )

        # State derivatives packaged for integrators
        position_dot = state.velocity_ned
        quaternion_dot = self._quaternion_derivative(state.quaternion_bn, omega)

        return position_dot, acceleration_ned, quaternion_dot, omega_dot

    @staticmethod
    def _quaternion_derivative(quat: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
        w, x, y, z = quat
        ox, oy, oz = omega_body
        return 0.5 * np.array(
            [
                -x * ox - y * oy - z * oz,
                w * ox + y * oz - z * oy,
                w * oy - x * oz + z * ox,
                w * oz + x * oy - y * ox,
            ]
        )

    def integrate(
        self,
        state: SixDofState,
        derivatives: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        dt: float,
    ) -> SixDofState:
        position_dot, accel_ned, quat_dot, omega_dot = derivatives
        next_state = state.copy()

        # Simple forward Euler integration with quaternion renormalization.
        next_state.position_ned += position_dot * dt
        next_state.velocity_ned += accel_ned * dt
        next_state.quaternion_bn += quat_dot * dt
        next_state.quaternion_bn = next_state.quaternion_bn / np.linalg.norm(
            next_state.quaternion_bn
        )
        next_state.angular_velocity_body += omega_dot * dt

        return next_state

    def rk4_step(
        self,
        state: SixDofState,
        thrust: float,
        moment_body: np.ndarray,
        env_gravity: np.ndarray,
        wind_ned: np.ndarray,
        external_force_body: np.ndarray,
        external_moment_body: np.ndarray,
        dt: float,
    ) -> SixDofState:
        """4th order Runge-Kutta"""

        def derivatives(local_state: SixDofState):
            return self.derivatives(
                local_state,
                thrust,
                moment_body,
                env_gravity,
                wind_ned,
                external_force_body,
                external_moment_body,
            )

        # RK4: evaluate derivatives at four points, blending for higher accuracy.
        k1 = self.derivatives(
            state,
            thrust,
            moment_body,
            env_gravity,
            wind_ned,
            external_force_body,
            external_moment_body,
        )

        state_k2 = state.copy()
        state_k2.position_ned += k1[0] * (dt * 0.5)
        state_k2.velocity_ned += k1[1] * (dt * 0.5)
        state_k2.quaternion_bn += k1[2] * (dt * 0.5)
        state_k2.quaternion_bn /= np.linalg.norm(state_k2.quaternion_bn)
        state_k2.angular_velocity_body += k1[3] * (dt * 0.5)
        k2 = self.derivatives(
            state_k2,
            thrust,
            moment_body,
            env_gravity,
            wind_ned,
            external_force_body,
            external_moment_body,
        )

        state_k3 = state.copy()
        state_k3.position_ned += k2[0] * (dt * 0.5)
        state_k3.velocity_ned += k2[1] * (dt * 0.5)
        state_k3.quaternion_bn += k2[2] * (dt * 0.5)
        state_k3.quaternion_bn /= np.linalg.norm(state_k3.quaternion_bn)
        state_k3.angular_velocity_body += k2[3] * (dt * 0.5)
        k3 = self.derivatives(
            state_k3,
            thrust,
            moment_body,
            env_gravity,
            wind_ned,
            external_force_body,
            external_moment_body,
        )

        state_k4 = state.copy()
        state_k4.position_ned += k3[0] * dt
        state_k4.velocity_ned += k3[1] * dt
        state_k4.quaternion_bn += k3[2] * dt
        state_k4.quaternion_bn /= np.linalg.norm(state_k4.quaternion_bn)
        state_k4.angular_velocity_body += k3[3] * dt
        k4 = self.derivatives(
            state_k4,
            thrust,
            moment_body,
            env_gravity,
            wind_ned,
            external_force_body,
            external_moment_body,
        )

        next_state = state.copy()
        next_state.position_ned += dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        next_state.velocity_ned += dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        next_state.quaternion_bn += dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        next_state.quaternion_bn /= np.linalg.norm(next_state.quaternion_bn)
        next_state.angular_velocity_body += (
            dt / 6.0 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        )
        return next_state
