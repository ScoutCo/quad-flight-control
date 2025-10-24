from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .commands import PositionVelocityCommand
from .math_utils import quat_to_euler
from .simulator import SimulationStepResult


class TelemetryLogger:
    """Minimal CSV logger for simulator telemetry."""

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
        "veh_roll",
        "veh_pitch",
        "veh_yaw",
        "cmd_roll",
        "cmd_pitch",
        "cmd_yaw",
        "thrust",
        "body_rate_command_x",
        "body_rate_command_y",
        "body_rate_command_z",
        "veh_body_rate_x",
        "veh_body_rate_y",
        "veh_body_rate_z",
    ]

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADERS)

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> TelemetryLogger:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def log(
        self,
        step: SimulationStepResult,
        command: PositionVelocityCommand,
        reference_position: Optional[Iterable[float]] = None,
        reference_velocity: Optional[Iterable[float]] = None,
    ) -> None:
        ref_pos = np.zeros(3) if reference_position is None else np.asarray(reference_position)
        ref_vel = np.zeros(3) if reference_velocity is None else np.asarray(reference_velocity)

        veh_roll, veh_pitch, veh_yaw = quat_to_euler(step.state.quaternion_bn)
        cmd_roll = cmd_pitch = cmd_yaw = 0.0
        if step.controller.attitude_target_quat is not None:
            cmd_roll, cmd_pitch, cmd_yaw = quat_to_euler(
                step.controller.attitude_target_quat
            )

        row = [
            step.time,
            *ref_pos.tolist(),
            *step.state.position_ned.tolist(),
            *ref_vel.tolist(),
            *step.state.velocity_ned.tolist(),
            *command.position_ned.tolist(),
            *command.velocity_ned_ff.tolist(),
            *command.accel_ned_ff.tolist(),
            veh_roll,
            veh_pitch,
            veh_yaw,
            cmd_roll,
            cmd_pitch,
            cmd_yaw,
            step.controller.thrust_command,
            *step.controller.body_rate_command.tolist(),
            *step.state.angular_velocity_body.tolist(),
        ]
        self._writer.writerow(row)
        self._file.flush()
