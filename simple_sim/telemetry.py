from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from sixdof_sim.commands import PositionVelocityCommand
from sixdof_sim.math_utils import quat_to_euler

from .simulator import SimpleSimulationStep


class SimpleTelemetryLogger:
    """CSV logger tailored for the simplified simulator state history."""

    HEADERS = [
        "time_s",
        "ref_pos_x",
        "ref_pos_y",
        "ref_pos_z",
        "veh_pos_x",
        "veh_pos_y",
        "veh_pos_z",
        "ref_vel_x",
        "ref_vel_y",
        "ref_vel_z",
        "veh_vel_x",
        "veh_vel_y",
        "veh_vel_z",
        "cmd_pos_x",
        "cmd_pos_y",
        "cmd_pos_z",
        "cmd_vel_x",
        "cmd_vel_y",
        "cmd_vel_z",
        "cmd_acc_x",
        "cmd_acc_y",
        "cmd_acc_z",
        "filt_acc_x",
        "filt_acc_y",
        "filt_acc_z",
        "veh_roll",
        "veh_pitch",
        "veh_yaw",
        "cmd_yaw",
    ]

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADERS)

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> SimpleTelemetryLogger:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def log(
        self,
        step: SimpleSimulationStep,
        command: PositionVelocityCommand,
        reference_position: Optional[Iterable[float]] = None,
        reference_velocity: Optional[Iterable[float]] = None,
    ) -> None:
        ref_pos = (
            np.zeros(3, dtype=float)
            if reference_position is None
            else np.asarray(reference_position, dtype=float)
        )
        ref_vel = (
            np.zeros(3, dtype=float)
            if reference_velocity is None
            else np.asarray(reference_velocity, dtype=float)
        )

        veh_roll, veh_pitch, veh_yaw = quat_to_euler(step.state.quaternion_bn)
        cmd_yaw = float(command.yaw_heading) if command.yaw_heading is not None else float("nan")

        row = [
            step.time_s,
            *ref_pos.tolist(),
            *step.state.position_ned.tolist(),
            *ref_vel.tolist(),
            *step.state.velocity_ned.tolist(),
            *command.position_ned.tolist(),
            *command.velocity_ned_ff.tolist(),
            *command.accel_ned_ff.tolist(),
            *step.filtered_accel_ned.tolist(),
            veh_roll,
            veh_pitch,
            veh_yaw,
            cmd_yaw,
        ]
        self._writer.writerow(row)
        self._file.flush()
