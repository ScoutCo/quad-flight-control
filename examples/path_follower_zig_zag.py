from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from examples import run_path_follower_example, zigzag_trajectory

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_FILE = LOG_DIR / "sim_path_follower_zig_zag.csv"
SIM_FINAL_TIME_S = 70.0


def main() -> None:
    trajectory = zigzag_trajectory(
        anchor_ned=(0.0, 0.0, -3.0),
        heading_deg=0.0,
        segment_length_m=40.0,
        num_segments=8,
        offset_angle_deg=40.0,
        speed_m_s=5.0,
        start_with_positive_offset=True,
    )

    run_path_follower_example(
        trajectory=trajectory,
        log_path=LOG_FILE,
        final_time_s=SIM_FINAL_TIME_S,
    )


if __name__ == "__main__":
    main()
