import numpy as np

from .config import PathFollowerConfig


class VelocitySmoother:
    """Filter velocity feed-forward commands while respecting slew limits."""

    def __init__(self, config: PathFollowerConfig):
        self._config = config
        self._last_velocity_command = np.zeros(3, dtype=float)
        self._last_command_time: float | None = None

    def reset(self) -> None:
        self._last_velocity_command = np.zeros(3, dtype=float)
        self._last_command_time = None

    def smooth(self, desired_velocity: np.ndarray, now_s: float) -> np.ndarray:
        v_desired = np.asarray(desired_velocity, dtype=float)
        if v_desired.shape != (3,):
            raise ValueError(
                f"desired_velocity must be length-3; received shape {v_desired.shape}"
            )

        if self._last_command_time is None:
            self._last_velocity_command = v_desired.copy()
            self._last_command_time = float(now_s)
            return v_desired.copy()

        dt = float(now_s) - self._last_command_time
        if dt < 0.0:
            # Negative time means clocks moved backwards; fall back to passthrough.
            self._last_velocity_command = v_desired.copy()
            self._last_command_time = float(now_s)
            return v_desired.copy()

        vel_prev = self._last_velocity_command

        def _exp_alpha(tau: float) -> float:
            return 1.0 if tau <= 0.0 else 1.0 - float(np.exp(-dt / tau))

        alpha_xy = _exp_alpha(self._config.tau_v_xy)
        alpha_z = _exp_alpha(self._config.tau_v_z)

        v_cmd = vel_prev.copy()
        v_cmd[0] = vel_prev[0] + alpha_xy * (v_desired[0] - vel_prev[0])
        v_cmd[1] = vel_prev[1] + alpha_xy * (v_desired[1] - vel_prev[1])
        v_cmd[2] = vel_prev[2] + alpha_z * (v_desired[2] - vel_prev[2])

        max_delta_xy = self._config.a_max_xy * dt
        delta_xy = v_cmd[:2] - vel_prev[:2]
        delta_xy_norm = float(np.linalg.norm(delta_xy))
        if delta_xy_norm > max_delta_xy > 0.0:
            scale = max_delta_xy / delta_xy_norm
            v_cmd[0] = vel_prev[0] + delta_xy[0] * scale
            v_cmd[1] = vel_prev[1] + delta_xy[1] * scale

        max_delta_z_up = self._config.a_max_up * dt
        max_delta_z_down = self._config.a_max_down * dt
        delta_z = v_cmd[2] - vel_prev[2]
        if delta_z > max_delta_z_up:
            v_cmd[2] = vel_prev[2] + max_delta_z_up
        elif delta_z < -max_delta_z_down:
            v_cmd[2] = vel_prev[2] - max_delta_z_down

        v_xy_norm = float(np.hypot(v_cmd[0], v_cmd[1]))
        if v_xy_norm > self._config.v_max_xy and v_xy_norm > 0.0:
            scale = self._config.v_max_xy / v_xy_norm
            v_cmd[0] *= scale
            v_cmd[1] *= scale

        v_cmd[2] = float(
            np.clip(v_cmd[2], -self._config.v_max_down, self._config.v_max_up)
        )

        self._last_velocity_command = v_cmd.copy()
        self._last_command_time = float(now_s)
        return v_cmd
