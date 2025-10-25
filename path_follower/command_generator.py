import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from sim import PositionVelocityCommand, State

from .path_follower import (
    PathFollowerDebug,
    PathFollowerException,
    PositionVelocityPathFollower,
)
from .plan import TrajectoryLike, build_plan

# Extend the plan far enough that lookahead checks remain valid between planner updates.
DEFAULT_PLAN_LOOKAHEAD_OFFSETS = (1.0, 2.0, 4.5)


@dataclass
class SimCommandGenerator:
    follower: PositionVelocityPathFollower
    trajectory: TrajectoryLike
    plan_offsets_s: Sequence[float]
    planner_delay_s: float
    plan_update_period_s: float
    last_plan_time_s: float = -math.inf
    last_command: PositionVelocityCommand | None = None
    last_debug: PathFollowerDebug | None = None
    last_plan_timestamp_s: float = float("nan")

    def __call__(self, time_s: float, state: State) -> PositionVelocityCommand:
        if (time_s - self.last_plan_time_s) >= self.plan_update_period_s - 1e-9:
            plan, plan_timestamp = build_plan(
                self.trajectory, self.plan_offsets_s, time_s, self.planner_delay_s
            )
            self.follower.handle_plan(plan)
            self.last_plan_time_s = time_s
            self.last_plan_timestamp_s = plan_timestamp

        try:
            result = self.follower.next_command(time_s)
        except PathFollowerException:
            if self.last_command is None:
                hold_position = state.position_ned.copy()
                self.last_command = PositionVelocityCommand(
                    position_ned=hold_position,
                    velocity_ned_ff=np.zeros(3),
                    yaw_heading=float(self.trajectory.yaw(time_s)),
                )
            self.last_debug = None
            return self.last_command

        self.last_command = result.command
        self.last_debug = result.debug
        return self.last_command
