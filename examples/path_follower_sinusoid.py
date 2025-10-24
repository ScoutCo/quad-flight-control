from __future__ import annotations

from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from examples import run_path_follower_example, sinusoid_trajectory

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_FILE = LOG_DIR / "sim_path_follower_sinusoid.csv"
SIM_FINAL_TIME_S = 30.0


def main() -> None:
    trajectory = sinusoid_trajectory(
        forward_speed_m_s=5.0,
        y_amplitude_m=5.0,
        y_frequency_hz=0.05,
        z_base_m=-5.0,
        z_amplitude_m=1.0,
        z_frequency_hz=0.025,
    )

    run_path_follower_example(
        trajectory=trajectory,
        log_path=LOG_FILE,
        final_time_s=SIM_FINAL_TIME_S,
    )


if __name__ == "__main__":
    main()
