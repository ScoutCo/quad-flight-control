import bisect
import math
from dataclasses import dataclass
from enum import Enum
from collections.abc import Sequence

import numpy as np

from quad_flight_control.sim import PositionVelocityCommand

from .config import PathFollowerConfig
from .plan import Plan, PlanState
from .velocity_smoother import VelocitySmoother


class PathFollowerError(Enum):
    NO_PLAN = "no_plan"
    EMPTY_PLAN = "empty_plan"
    PLAN_NUM_STATES = "plan_num_states"
    PLAN_IN_FUTURE = "plan_in_future"
    PLAN_LENGTH = "plan_length"
    PLAN_AGE = "plan_age"


class PathFollowerException(RuntimeError):
    def __init__(self, error: PathFollowerError, message: str):
        super().__init__(message)
        self.error = error


@dataclass(frozen=True)
class PathFollowerDebug:
    interpolated_state: PlanState
    feedforward_velocity_ned: np.ndarray
    smoothed_velocity_ned: np.ndarray


@dataclass(frozen=True)
class PathFollowerResult:
    command: PositionVelocityCommand
    debug: PathFollowerDebug


class PositionVelocityPathFollower:
    """Interpolate a plan and produce position/velocity setpoints."""

    def __init__(
        self,
        config: PathFollowerConfig | None = None,
        velocity_smoother: VelocitySmoother | None = None,
    ):
        self._config = config or PathFollowerConfig()
        self._velocity_smoother = velocity_smoother or VelocitySmoother(self._config)
        self._plan: Plan | None = None

    def handle_plan(self, plan: Plan) -> None:
        self._plan = plan

    def clear_plan(self) -> None:
        self._plan = None

    def reset(self) -> None:
        self.clear_plan()
        self._velocity_smoother.reset()

    @property
    def has_plan(self) -> bool:
        return self._plan is not None

    def next_command(self, now_s: float) -> PathFollowerResult:
        plan = self._plan
        if plan is None:
            raise PathFollowerException(
                PathFollowerError.NO_PLAN, "No plan has been received."
            )

        states = plan.states
        if not states:
            raise PathFollowerException(
                PathFollowerError.EMPTY_PLAN, "Plan contains no states."
            )

        if len(states) < 2:
            raise PathFollowerException(
                PathFollowerError.PLAN_NUM_STATES,
                "Plan must contain at least two states to infer velocity.",
            )

        plan_age = float(now_s) - plan.timestamp_s
        if plan_age > self._config.max_plan_age_s:
            raise PathFollowerException(
                PathFollowerError.PLAN_AGE,
                f"Plan is too old ({plan_age:.3f} s).",
            )

        max_future = self._config.max_plan_future_s
        if (states[0].time_s - now_s) > max_future:
            raise PathFollowerException(
                PathFollowerError.PLAN_IN_FUTURE,
                "Plan start time is too far in the future relative to now.",
            )

        lookahead_time = now_s + self._config.lookahead_offset_s
        if lookahead_time - states[-1].time_s > 0.0:
            raise PathFollowerException(
                PathFollowerError.PLAN_LENGTH,
                "Plan does not extend far enough into the future.",
            )

        interp_state = _interpolate_plan_state(states, now_s)
        vel_ff = _compute_feedforward_velocity(states, lookahead_time)
        vel_cmd = self._velocity_smoother.smooth(vel_ff, now_s)

        command = PositionVelocityCommand(
            position_ned=interp_state.position_ned.copy(),
            velocity_ned_ff=vel_cmd.copy(),
            yaw_heading=interp_state.yaw_rad,
        )

        debug = PathFollowerDebug(
            interpolated_state=interp_state,
            feedforward_velocity_ned=vel_ff.copy(),
            smoothed_velocity_ned=vel_cmd.copy(),
        )
        return PathFollowerResult(command=command, debug=debug)


def _get_lower_upper_state_bounds(
    states: Sequence[PlanState], lookup_time_s: float
) -> tuple[PlanState, PlanState]:
    times = [state.time_s for state in states]
    upper_idx = bisect.bisect_left(times, lookup_time_s)
    if upper_idx >= len(states):
        upper_idx = len(states) - 1
        lower_idx = upper_idx - 1
    elif upper_idx == 0:
        lower_idx = 0
        upper_idx = 1
    else:
        lower_idx = upper_idx - 1
    return states[lower_idx], states[upper_idx]


def _interpolate_plan_state(
    states: Sequence[PlanState], interp_time_s: float
) -> PlanState:
    lower, upper = _get_lower_upper_state_bounds(states, interp_time_s)
    dt = upper.time_s - lower.time_s
    if dt > 0.0:
        alpha = max(0.0, min(1.0, (interp_time_s - lower.time_s) / dt))
    else:
        alpha = 0.0

    position = lower.position_ned + alpha * (upper.position_ned - lower.position_ned)
    yaw = _lerp_angle(lower.yaw_rad, upper.yaw_rad, alpha)

    return PlanState(position_ned=position, yaw_rad=yaw, time_s=interp_time_s)


def _lerp_angle(a: float, b: float, alpha: float) -> float:
    diff = math.remainder(b - a, 2.0 * math.pi)
    return a + alpha * diff


def _compute_feedforward_velocity(
    states: Sequence[PlanState], lookup_time_s: float
) -> np.ndarray:
    lower, upper = _get_lower_upper_state_bounds(states, lookup_time_s)
    dt = upper.time_s - lower.time_s
    if dt <= 0.0:
        return np.zeros(3, dtype=float)
    delta = upper.position_ned - lower.position_ned
    return delta / dt
