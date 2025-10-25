from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from examples import circle_trajectory, run_path_follower_example

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_FILE = LOG_DIR / "sim_path_follower_circle.csv"
SIM_FINAL_TIME_S = 32.0


def main() -> None:
    trajectory = circle_trajectory(
        radius_m=10.0,
        altitude_m=-5.0,
        period_s=60.0,
        phase_rad=0.0,
    )

    run_path_follower_example(
        trajectory=trajectory,
        log_path=LOG_FILE,
        final_time_s=SIM_FINAL_TIME_S,
    )


if __name__ == "__main__":
    main()
