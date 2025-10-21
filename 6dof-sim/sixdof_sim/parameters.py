from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class PositionGains:
    kp: np.ndarray = field(default_factory=lambda: np.array([1.5, 1.5, 1.8]))
    ki: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.1, 0.2]))
    kd: np.ndarray = field(default_factory=lambda: np.array([0.6, 0.6, 0.8]))
    integrator_limit: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0, 2.0]))


@dataclass
class VelocityGains:
    kp: np.ndarray = field(default_factory=lambda: np.array([1.8, 1.8, 2.5]))
    ki: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.2, 0.4]))
    kd: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.4, 0.6]))
    integrator_limit: np.ndarray = field(default_factory=lambda: np.array([3.0, 3.0, 3.0]))


@dataclass
class AttitudeGains:
    kp: float = 5.0


@dataclass
class RateGains:
    kp: np.ndarray = field(default_factory=lambda: np.array([0.12, 0.12, 0.1]))
    ki: np.ndarray = field(default_factory=lambda: np.array([0.02, 0.02, 0.01]))
    kd: np.ndarray = field(default_factory=lambda: np.array([0.003, 0.003, 0.002]))
    integrator_limit: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 0.6]))


@dataclass
class ControllerGains:
    position: PositionGains = field(default_factory=PositionGains)
    velocity: VelocityGains = field(default_factory=VelocityGains)
    attitude: AttitudeGains = field(default_factory=AttitudeGains)
    rate: RateGains = field(default_factory=RateGains)
    position_lag_time_constant: float = 0.25
    velocity_lag_time_constant: float = 0.15
    attitude_lag_time_constant: float = 0.08
    rate_lag_time_constant: float = 0.03


@dataclass
class ActuatorParameters:
    thrust_time_constant: float = 0.15
    moment_time_constant: float = 0.07
    max_thrust: float = 25.0
    max_moment: np.ndarray = field(default_factory=lambda: np.array([2.5, 2.5, 1.8]))
    thrust_min: float = 0.0


@dataclass
class VehicleParameters:
    mass: float = 1.5
    inertia: np.ndarray = field(default_factory=lambda: np.diag([0.03, 0.03, 0.05]))
    drag_coeff_linear: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.1, 0.2]))
    angular_damping: np.ndarray = field(default_factory=lambda: np.array([0.02, 0.02, 0.04]))
    disk_loading: float = 4.0

    def inverse_inertia(self) -> np.ndarray:
        return np.linalg.inv(self.inertia)


@dataclass
class EnvironmentParameters:
    gravity: float = 9.80665
    wind_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))
    disturbance_force_body: np.ndarray = field(default_factory=lambda: np.zeros(3))
    disturbance_moment_body: np.ndarray = field(default_factory=lambda: np.zeros(3))
