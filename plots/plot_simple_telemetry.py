from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot telemetry captured from the simple simulator."
    )
    parser.add_argument("logfile", type=Path, help="Path to a simple sim CSV log")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional path to save the figure instead of displaying it.",
    )
    return parser


def plot_simple_telemetry(df: pd.DataFrame, output: Path | None) -> None:
    time = df["time_s"].to_numpy()

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    for comp, label in zip(("x", "y", "z"), ("X", "Y", "Z")):
        axes[0].plot(time, df[f"veh_pos_{comp}"], label=f"{label} actual")
        axes[0].plot(
            time,
            df[f"ref_pos_{comp}"],
            linestyle="--",
            label=f"{label} reference",
        )
    axes[0].set_ylabel("Position (m)")
    axes[0].legend(loc="upper right", fontsize="small")
    axes[0].grid(True, linestyle=":")

    for comp, label in zip(("x", "y", "z"), ("X", "Y", "Z")):
        axes[1].plot(time, df[f"veh_vel_{comp}"], label=f"{label} actual")
        axes[1].plot(
            time,
            df[f"ref_vel_{comp}"],
            linestyle="--",
            label=f"{label} reference",
        )
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].legend(loc="upper right", fontsize="small")
    axes[1].grid(True, linestyle=":")

    for comp, label in zip(("x", "y", "z"), ("X", "Y", "Z")):
        axes[2].plot(time, df[f"filt_acc_{comp}"], label=f"{label} filtered")
        axes[2].plot(
            time,
            df[f"cmd_acc_{comp}"],
            linestyle="--",
            label=f"{label} command",
        )
    axes[2].set_ylabel("Acceleration (m/s²)")
    axes[2].legend(loc="upper right", fontsize="small")
    axes[2].grid(True, linestyle=":")

    axes[3].plot(time, np.degrees(df["veh_roll"]), label="Roll (deg)")
    axes[3].plot(time, np.degrees(df["veh_pitch"]), label="Pitch (deg)")
    axes[3].plot(time, np.degrees(df["veh_yaw"]), label="Yaw (deg)")
    if "cmd_yaw" in df.columns and df["cmd_yaw"].notna().any():
        axes[3].plot(
            time,
            np.degrees(df["cmd_yaw"].fillna(method="ffill")),
            linestyle="--",
            label="Yaw cmd (deg)",
        )
    axes[3].set_ylabel("Attitude (deg)")
    axes[3].legend(loc="upper right", fontsize="small")
    axes[3].grid(True, linestyle=":")

    axes[3].set_xlabel("Time (s)")

    fig.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
    else:
        plt.show()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.logfile.exists():
        raise SystemExit(f"Telemetry log not found: {args.logfile}")

    df = pd.read_csv(args.logfile)
    missing = [col for col in ("veh_pos_x", "veh_vel_x", "filt_acc_x") if col not in df]
    if missing:
        raise SystemExit(f"Log is missing expected columns: {missing}")

    plot_simple_telemetry(df, args.output)


if __name__ == "__main__":
    main()
