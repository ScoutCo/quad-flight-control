from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from examples import line_trajectory, run_path_follower_example

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_FILE = LOG_DIR / "sim_path_follower_line.csv"
SIM_FINAL_TIME_S = 25.0


def main() -> None:
    trajectory = line_trajectory(
        speed_m_s=4.5,
        heading_rad=0.1,
        altitude_m=-2.0,
        lateral_offset_m=0.0,
    )

    run_path_follower_example(
        trajectory=trajectory,
        log_path=LOG_FILE,
        final_time_s=SIM_FINAL_TIME_S,
    )


if __name__ == "__main__":
    main()
