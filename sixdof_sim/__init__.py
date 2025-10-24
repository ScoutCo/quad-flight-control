"""Lightweight 6DOF quadrotor flight dynamics simulator."""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from .states import SixDofState
from .parameters import (
    VehicleParameters,
    ControllerGains,
    PositionGains,
    VelocityGains,
    AttitudeGains,
    RateGains,
    EnvironmentParameters,
    ActuatorParameters,
)
from .commands import PositionVelocityCommand
from .simulator import SixDofSimulator

__all__ = [
    "SixDofState",
    "VehicleParameters",
    "ControllerGains",
    "PositionGains",
    "VelocityGains",
    "AttitudeGains",
    "RateGains",
    "EnvironmentParameters",
    "ActuatorParameters",
    "PositionVelocityCommand",
    "SixDofSimulator",
    "PathFollowerConfig",
    "PathFollowerDebug",
    "PathFollowerError",
    "PathFollowerException",
    "PathFollowerResult",
    "Plan",
    "PlanState",
    "PositionVelocityPathFollower",
    "VelocitySmoother",
]

_OFFBOARD_EXPORTS = {
    "PathFollowerConfig",
    "PathFollowerDebug",
    "PathFollowerError",
    "PathFollowerException",
    "PathFollowerResult",
    "Plan",
    "PlanState",
    "PositionVelocityPathFollower",
    "VelocitySmoother",
}

if TYPE_CHECKING:
    from offboard_control import (
        PathFollowerConfig,
        PathFollowerDebug,
        PathFollowerError,
        PathFollowerException,
        PathFollowerResult,
        Plan,
        PlanState,
        PositionVelocityPathFollower,
        VelocitySmoother,
    )


def __getattr__(name: str):  # pragma: no cover - passthrough
    if name in _OFFBOARD_EXPORTS:
        module = import_module("offboard_control")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - passthrough
    return sorted(set(globals()) | _OFFBOARD_EXPORTS)
