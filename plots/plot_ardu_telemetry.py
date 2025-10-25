import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_log(path: Path) -> None:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        raise ValueError(f"Log {path} is empty")
    data = np.atleast_1d(data)

    t = data["time_s"]
    ref_pos = np.vstack([data["ref_pos_x"], data["ref_pos_y"], data["ref_pos_z"]]).T
    veh_pos = np.vstack([data["veh_pos_x"], data["veh_pos_y"], data["veh_pos_z"]]).T
    ref_vel = np.vstack([data["ref_vel_x"], data["ref_vel_y"], data["ref_vel_z"]]).T
    veh_vel = np.vstack([data["veh_vel_x"], data["veh_vel_y"], data["veh_vel_z"]]).T
    cmd_vel = np.vstack([data["cmd_vel_x"], data["cmd_vel_y"], data["cmd_vel_z"]]).T
    cmd_acc = np.vstack([data["cmd_acc_x"], data["cmd_acc_y"], data["cmd_acc_z"]]).T
    thrust = data["thrust"]

    fig, axes = plt.subplots(4, 1, figsize=(11, 14), sharex=True)

    axes[0].plot(t, ref_pos[:, 0], label="ref_x")
    axes[0].plot(t, ref_pos[:, 1], label="ref_y")
    axes[0].plot(t, ref_pos[:, 2], label="ref_z")
    axes[0].plot(t, veh_pos[:, 0], "--", label="veh_x")
    axes[0].plot(t, veh_pos[:, 1], "--", label="veh_y")
    axes[0].plot(t, veh_pos[:, 2], "--", label="veh_z")
    axes[0].set_ylabel("Position [m]")
    axes[0].legend(loc="upper right", ncol=2)

    axes[1].plot(t, ref_vel[:, 0], label="ref_vx")
    axes[1].plot(t, ref_vel[:, 1], label="ref_vy")
    axes[1].plot(t, ref_vel[:, 2], label="ref_vz")
    axes[1].plot(t, veh_vel[:, 0], "--", label="veh_vx")
    axes[1].plot(t, veh_vel[:, 1], "--", label="veh_vy")
    axes[1].plot(t, veh_vel[:, 2], "--", label="veh_vz")
    axes[1].plot(t, cmd_vel[:, 0], ":", label="cmd_vx")
    axes[1].plot(t, cmd_vel[:, 1], ":", label="cmd_vy")
    axes[1].plot(t, cmd_vel[:, 2], ":", label="cmd_vz")
    axes[1].set_ylabel("Velocity [m/s]")
    axes[1].legend(loc="upper right", ncol=3)

    axes[2].plot(t, cmd_acc[:, 0], label="cmd_ax")
    axes[2].plot(t, cmd_acc[:, 1], label="cmd_ay")
    axes[2].plot(t, cmd_acc[:, 2], label="cmd_az")
    axes[2].set_ylabel("Acceleration [m/s²]")
    axes[2].legend(loc="upper right")

    axes[3].plot(t, thrust, label="thrust")
    axes[3].set_ylabel("Thrust [N]")
    axes[3].legend(loc="upper right")
    axes[3].set_xlabel("Time [s]")

    fig.suptitle(f"Path follower telemetry: {path.name}")
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot path follower telemetry log")
    parser.add_argument("log_path", help="CSV log file produced by TelemetryLogger")
    args = parser.parse_args()
    plot_log(Path(args.log_path))


if __name__ == "__main__":
    main()
