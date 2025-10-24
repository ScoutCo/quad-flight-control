from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PlanState:
    position_ned: np.ndarray
    yaw_rad: float
    time_s: float

    def __post_init__(self) -> None:
        position = np.asarray(self.position_ned, dtype=float)
        if position.shape != (3,):
            raise ValueError(
                f"PlanState.position_ned must be length-3; received shape {position.shape}"
            )
        object.__setattr__(self, "position_ned", position.copy())
        object.__setattr__(self, "yaw_rad", float(self.yaw_rad))
        object.__setattr__(self, "time_s", float(self.time_s))


@dataclass(frozen=True)
class Plan:
    states: Tuple[PlanState, ...]
    timestamp_s: float
    frame_id: str | None = None

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("Plan must contain at least one state.")
        sorted_states = tuple(sorted(self.states, key=lambda state: state.time_s))
        if any(
            later.time_s < earlier.time_s
            for earlier, later in zip(sorted_states, sorted_states[1:])
        ):
            raise ValueError("Plan state times must be non-decreasing.")
        object.__setattr__(self, "states", sorted_states)
        object.__setattr__(self, "timestamp_s", float(self.timestamp_s))

    def __len__(self) -> int:
        return len(self.states)

    def __iter__(self) -> Iterator[PlanState]:
        return iter(self.states)

    @property
    def start_time_s(self) -> float:
        return self.states[0].time_s

    @property
    def end_time_s(self) -> float:
        return self.states[-1].time_s

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s

    def copy_with_states(self, states: Iterable[PlanState]) -> Plan:
        """Return a shallow copy with a new set of states."""
        return Plan(tuple(states), self.timestamp_s, self.frame_id)


class TrajectoryLike(Protocol):
    def position(self, time_s: float) -> np.ndarray: ...

    def velocity(self, time_s: float) -> np.ndarray: ...

    def yaw(self, time_s: float) -> float: ...


def build_plan(
    trajectory: TrajectoryLike,
    plan_offsets_s: Sequence[float],
    now_s: float,
    planner_delay_s: float,
) -> tuple[Plan, float]:
    plan_timestamp = now_s - planner_delay_s
    states: tuple[PlanState, ...] = tuple(
        PlanState(
            position_ned=trajectory.position(plan_timestamp + offset),
            yaw_rad=trajectory.yaw(plan_timestamp + offset),
            time_s=plan_timestamp + offset,
        )
        for offset in plan_offsets_s
    )
    if len(states) < 2:
        raise ValueError("Path follower requires at least two plan states.")
    return Plan(states=states, timestamp_s=plan_timestamp, frame_id="ref_ned"), plan_timestamp
