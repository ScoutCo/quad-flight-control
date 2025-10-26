import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.widgets import Button, Slider  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


def find_csv(directory: Path, suffix: str) -> Path:
    matches = sorted(directory.glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"No CSV ending with '{suffix}' found in {directory}")
    if len(matches) > 1:
        raise FileExistsError(
            f"Multiple CSV files ending with '{suffix}' found in {directory}: {matches}"
        )
    return matches[0]


def load_dataframes(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    control_path = find_csv(directory, "_control.csv")
    plan_path = find_csv(directory, "_plan.csv")

    control_df = pd.read_csv(control_path)
    if "time_s" in control_df.columns:
        control_df = control_df.set_index("time_s")

    plan_df = pd.read_csv(plan_path)
    if "plan_time_s" in plan_df.columns:
        plan_df = plan_df.set_index("plan_time_s")

    return control_df, plan_df


def get_run_sequence(control_df: pd.DataFrame) -> list[int]:
    run_series = control_df.get("autonomy_run_idx")
    if run_series is None:
        return []
    runs = run_series.dropna().unique()
    runs = sorted(int(r) for r in runs if np.isfinite(r))
    return runs


def extract_run_slice(control_df: pd.DataFrame, run_id: int) -> pd.DataFrame:
    if "autonomy_run_idx" not in control_df:
        return control_df.iloc[0:0]
    return control_df[control_df["autonomy_run_idx"] == run_id]


def _extract_time_axis(control_df: pd.DataFrame) -> np.ndarray:
    index = control_df.index
    if isinstance(index, pd.Index) and np.issubdtype(index.dtype, np.number):
        return index.to_numpy(dtype=float, copy=False)
    for candidate in ("time_s", "time", "t_s"):
        if candidate in control_df.columns:
            return control_df[candidate].to_numpy(dtype=float, copy=False)
    return np.arange(len(control_df), dtype=float)


def _yaw_from_quaternion(x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def _draw_plan_arrow(
    ax,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
) -> Optional[object]:
    if xs.size < 2:
        return None
    arrows: list[object] = []
    for idx in range(1, xs.size):
        tail = np.array([xs[idx - 1], ys[idx - 1], zs[idx - 1]], dtype=float)
        head = np.array([xs[idx], ys[idx], zs[idx]], dtype=float)
        if not np.all(np.isfinite(head)) or not np.all(np.isfinite(tail)):
            continue
        direction = head - tail
        norm = np.linalg.norm(direction)
        if norm <= 0.0 or not np.isfinite(norm):
            continue
        unit_dir = direction / norm
        arrow_length = min(norm * 0.2, 0.4)
        start_point = head - unit_dir * arrow_length
        arrow = ax.quiver(
            start_point[0],
            start_point[1],
            start_point[2],
            unit_dir[0],
            unit_dir[1],
            unit_dir[2],
            length=arrow_length,
            color="tab:green",
            alpha=0.9,
            arrow_length_ratio=0.35,
            linewidth=1.0,
        )
        arrows.append(arrow)
    return arrows


def plot_tracking_summary(control_df: pd.DataFrame) -> plt.Figure | None:
    if control_df.empty:
        print("Control dataframe is empty; skipping tracking summary plot.")
        return None

    required_columns = [
        "odom_pos_x",
        "odom_pos_y",
        "odom_pos_z",
        "cmd_pos_x",
        "cmd_pos_y",
        "cmd_pos_z",
        "cmd_yaw_rad",
        "odom_quat_x",
        "odom_quat_y",
        "odom_quat_z",
        "odom_quat_w",
        "autonomy_run_idx",
    ]
    missing_columns = [col for col in required_columns if col not in control_df.columns]
    if missing_columns:
        raise KeyError(f"Missing expected columns in control dataframe: {missing_columns}")

    time = _extract_time_axis(control_df)
    fig, (ax_pos, ax_yaw, ax_run) = plt.subplots(
        3,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 0.6]},
    )
    fig.suptitle("Control vs Odometry Tracking")

    colors = {"x": "tab:red", "y": "tab:green", "z": "tab:blue"}
    for component in ("x", "y", "z"):
        ax_pos.plot(
            time,
            control_df[f"odom_pos_{component}"].to_numpy(dtype=float, copy=False),
            label=f"odom_{component}",
            color=colors[component],
            linestyle="-",
        )
        ax_pos.plot(
            time,
            control_df[f"cmd_pos_{component}"].to_numpy(dtype=float, copy=False),
            label=f"cmd_{component}",
            color=colors[component],
            linestyle="--",
        )

    ax_pos.set_ylabel("Position (m)")
    ax_pos.legend(loc="upper right", ncol=3, fontsize="small")
    ax_pos.set_title("Position Tracking")

    actual_yaw_rad = _yaw_from_quaternion(
        control_df["odom_quat_x"].to_numpy(dtype=float, copy=False),
        control_df["odom_quat_y"].to_numpy(dtype=float, copy=False),
        control_df["odom_quat_z"].to_numpy(dtype=float, copy=False),
        control_df["odom_quat_w"].to_numpy(dtype=float, copy=False),
    )
    commanded_yaw_rad = control_df["cmd_yaw_rad"].to_numpy(dtype=float, copy=False)

    ax_yaw.plot(
        time,
        np.degrees(np.unwrap(actual_yaw_rad)),
        label="actual yaw",
        color="tab:purple",
    )
    ax_yaw.plot(
        time,
        np.degrees(np.unwrap(commanded_yaw_rad)),
        label="commanded yaw",
        color="tab:orange",
        linestyle="--",
    )

    ax_yaw.set_ylabel("Yaw (deg)")
    ax_yaw.legend(loc="upper right", fontsize="small")
    ax_yaw.set_title("Yaw Tracking")

    run_indices = control_df["autonomy_run_idx"].to_numpy(dtype=float, copy=False)
    if np.isfinite(run_indices).any():
        ax_run.step(
            time,
            run_indices,
            where="post",
            color="tab:brown",
            label="autonomy run",
        )
        finite_runs = np.unique(run_indices[np.isfinite(run_indices)])
        ax_run.set_yticks(finite_runs)
    else:
        ax_run.plot(time, run_indices, color="tab:brown", label="autonomy run")
        ax_run.set_yticks([])

    ax_run.set_ylabel("Run")
    ax_run.set_xlabel("Time (s)")
    ax_run.set_title("Run Index")
    ax_run.legend(loc="upper right", fontsize="small")

    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig


def plot_runs(
    control_df: pd.DataFrame,
    plan_df: pd.DataFrame | None,
    runs: list[int],
) -> plt.Figure | None:
    if control_df.empty:
        print("Control dataframe is empty; skipping 3D path plot.")
        return None
    position_columns = [
        "odom_pos_x",
        "odom_pos_y",
        "odom_pos_z",
        "cmd_pos_x",
        "cmd_pos_y",
        "cmd_pos_z",
    ]
    missing = [col for col in position_columns if col not in control_df.columns]
    if missing:
        raise KeyError(f"Missing expected position columns for 3D plot: {missing}")

    plan_available = plan_df is not None and not plan_df.empty
    step_indices: list[int] = []
    if plan_available:
        if "autonomy_run_idx" not in plan_df.columns:
            raise KeyError("Plan dataframe is missing 'autonomy_run_idx'")
        step_indices = sorted(
            {
                int(col.split("_")[0][4:])
                for col in plan_df.columns
                if col.startswith("step") and col.endswith("_x")
            }
        )
        if not step_indices:
            plan_available = False

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    line_odom = ax.plot([], [], [], lw=2, color="tab:blue", label="odom")[0]
    line_cmd = ax.plot([], [], [], lw=2, color="tab:orange", linestyle="--", label="cmd")[0]
    point_marker = ax.scatter(
        [],
        [],
        [],
        s=120,
        color="tab:blue",
        edgecolor="k",
        linewidth=0.5,
        alpha=0.9,
        depthshade=False,
        label="current",
    )
    plan_proxy = Line2D([0], [0], color="tab:green", linestyle="-", alpha=0.6, label="plan")
    legend_handles = [line_odom, line_cmd, point_marker] + ([plan_proxy] if plan_available else [])
    ax.legend(handles=legend_handles, loc="best")
    title = ax.set_title("Autonomy Run")

    run_state: dict[str, object] = {
        "index": 0,
        "runs": runs,
        "current_run_id": None,
        "position_index": 0,
        "odom_coords": (np.array([], dtype=float),) * 3,
        "odom_times": np.array([], dtype=float),
        "cmd_coords": (np.array([], dtype=float),) * 3,
        "cmd_times": np.array([], dtype=float),
        "plan_entries": [],
        "title": "All samples",
        "position_count": 0,
        "slider": None,
        "suppress_slider_callback": False,
    }
    plan_lines: list = []
    plan_arrows: list[list[object]] = []

    def update_plot(run_idx: Optional[int] = None) -> None:
        nonlocal plan_lines, plan_arrows
        run_sequence = run_state["runs"]
        label = "All samples"
        if run_sequence:
            if run_idx is None:
                run_idx = run_sequence[run_state["index"]]
            if run_idx != run_state.get("current_run_id"):
                run_state["position_index"] = 0
            run_slice = extract_run_slice(control_df, run_idx)
            if run_slice.empty:
                run_slice = control_df
                print(
                    f"Warning: autonomy run {run_idx} produced no rows; plotting full dataset instead."
                )
            else:
                label = f"Autonomy Run {run_idx}"
            run_state["current_run_id"] = run_idx
        else:
            run_slice = control_df
            btn_prev.set_visible(False)
            btn_next.set_visible(False)
            run_state["current_run_id"] = None

        plan_entries: list[dict[str, np.ndarray | float]] = []
        if plan_available and run_sequence:
            plan_slice = plan_df[plan_df["autonomy_run_idx"] == run_idx]
        elif plan_available:
            plan_slice = plan_df
        else:
            plan_slice = None

        odom_x = run_slice["odom_pos_x"].to_numpy(dtype=float, copy=False)
        odom_y = run_slice["odom_pos_y"].to_numpy(dtype=float, copy=False)
        odom_z = run_slice["odom_pos_z"].to_numpy(dtype=float, copy=False)
        cmd_x = run_slice["cmd_pos_x"].to_numpy(dtype=float, copy=False)
        cmd_y = run_slice["cmd_pos_y"].to_numpy(dtype=float, copy=False)
        cmd_z = run_slice["cmd_pos_z"].to_numpy(dtype=float, copy=False)

        odom_mask = np.isfinite(odom_x) & np.isfinite(odom_y) & np.isfinite(odom_z)
        cmd_mask = np.isfinite(cmd_x) & np.isfinite(cmd_y) & np.isfinite(cmd_z)
        odom_indices = np.where(odom_mask)[0]
        cmd_indices = np.where(cmd_mask)[0]

        if odom_indices.size:
            odom_x_valid = odom_x[odom_indices]
            odom_y_valid = odom_y[odom_indices]
            odom_z_valid = odom_z[odom_indices]
        else:
            odom_x_valid = odom_y_valid = odom_z_valid = np.array([], dtype=float)

        if cmd_indices.size:
            cmd_x_valid = cmd_x[cmd_indices]
            cmd_y_valid = cmd_y[cmd_indices]
            cmd_z_valid = cmd_z[cmd_indices]
        else:
            cmd_x_valid = cmd_y_valid = cmd_z_valid = np.array([], dtype=float)

        # Clear previous plan lines
        for line in plan_lines:
            line.remove()
        plan_lines = []
        for arrow_group in plan_arrows:
            for arrow in arrow_group:
                arrow.remove()
        plan_arrows = []

        plan_bounds: list[np.ndarray] = []
        if plan_slice is not None and not plan_slice.empty:
            plan_times = plan_slice.index.to_numpy(dtype=float, copy=False)
            plan_x_cols = [f"step{idx}_x" for idx in step_indices]
            plan_y_cols = [f"step{idx}_y" for idx in step_indices]
            plan_z_cols = [f"step{idx}_z" for idx in step_indices]
            plan_x_values = plan_slice[plan_x_cols].to_numpy(dtype=float, copy=False)
            plan_y_values = plan_slice[plan_y_cols].to_numpy(dtype=float, copy=False)
            plan_z_values = plan_slice[plan_z_cols].to_numpy(dtype=float, copy=False)

            for idx_row, (px, py, pz) in enumerate(
                zip(plan_x_values, plan_y_values, plan_z_values)
            ):
                valid = np.isfinite(px) & np.isfinite(py) & np.isfinite(pz)
                if not np.any(valid):
                    continue
                px_valid = px[valid]
                py_valid = py[valid]
                pz_valid = pz[valid]
                plan_entries.append(
                    {
                        "time": float(plan_times[idx_row]) if idx_row < plan_times.size else float("nan"),
                        "x": px_valid,
                        "y": py_valid,
                        "z": pz_valid,
                    }
                )
                plan_bounds.append(np.column_stack((px_valid, py_valid, pz_valid)))

        run_state["plan_entries"] = plan_entries
        run_state["odom_coords"] = (odom_x_valid, odom_y_valid, odom_z_valid)
        run_state["cmd_coords"] = (cmd_x_valid, cmd_y_valid, cmd_z_valid)
        run_state["position_count"] = int(odom_indices.size)
        time_values = _extract_time_axis(run_slice)
        run_state["odom_times"] = (
            time_values[odom_indices] if odom_indices.size else np.array([], dtype=float)
        )
        run_state["cmd_times"] = (
            time_values[cmd_indices] if cmd_indices.size else np.array([], dtype=float)
        )
        run_state["title"] = label
        if run_state["position_count"] == 0:
            run_state["position_index"] = 0
        else:
            run_state["position_index"] = min(
                int(run_state.get("position_index", 0)), run_state["position_count"] - 1
            )

        bounds_components = []
        if odom_x_valid.size:
            bounds_components.append(
                np.column_stack((odom_x_valid, odom_y_valid, odom_z_valid))
            )
        if cmd_x_valid.size:
            bounds_components.append(np.column_stack((cmd_x_valid, cmd_y_valid, cmd_z_valid)))
        bounds_components.extend(plan_bounds)
        combined = (
            np.vstack(bounds_components)
            if bounds_components
            else np.zeros((1, 3), dtype=float)
        )
        data_min = np.array(
            [
                float(np.min(combined[:, 0])),
                float(np.min(combined[:, 1])),
                float(np.min(combined[:, 2])),
            ],
            dtype=float,
        )
        data_max = np.array(
            [
                float(np.max(combined[:, 0])),
                float(np.max(combined[:, 1])),
                float(np.max(combined[:, 2])),
            ],
            dtype=float,
        )
        center = 0.5 * (data_max + data_min)
        half_range = 0.5 * np.max(data_max - data_min)
        if not np.isfinite(half_range) or half_range <= 0.0:
            half_range = 1.0
        lower = center - half_range
        upper = center + half_range
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_zlim(upper[2], lower[2])  # In NED frame, smaller (more negative) values are higher altitude
        try:
            ax.set_box_aspect((1.0, 1.0, 1.0))
        except AttributeError:
            pass
        _update_slider_state()
        update_position_marker()
        fig.canvas.draw_idle()

    def _current_time() -> float | None:
        times = run_state.get("odom_times")
        idx = int(run_state.get("position_index", 0))
        if isinstance(times, np.ndarray) and times.size > idx:
            value = float(times[idx])
            if np.isfinite(value):
                return value
        return None

    def _update_control_history() -> None:
        coords = run_state.get("odom_coords")
        idx = int(run_state.get("position_index", 0))
        if (
            isinstance(coords, tuple)
            and len(coords) == 3
            and all(isinstance(arr, np.ndarray) for arr in coords)
        ):
            xs, ys, zs = coords
            upto = min(idx + 1, xs.size)
            line_odom.set_data(xs[:upto], ys[:upto])
            line_odom.set_3d_properties(zs[:upto])
        else:
            line_odom.set_data([], [])
            line_odom.set_3d_properties([])

        cmd_coords = run_state.get("cmd_coords")
        cmd_times = run_state.get("cmd_times")
        current_time = _current_time()
        if (
            isinstance(cmd_coords, tuple)
            and len(cmd_coords) == 3
            and all(isinstance(arr, np.ndarray) for arr in cmd_coords)
        ):
            xs, ys, zs = cmd_coords
            if xs.size == 0:
                line_cmd.set_data([], [])
                line_cmd.set_3d_properties([])
            else:
                if isinstance(cmd_times, np.ndarray) and cmd_times.size == xs.size and current_time is not None:
                    cutoff = int(np.searchsorted(cmd_times, current_time + 1e-9, side="right"))
                else:
                    cutoff = min(idx + 1, xs.size)
                cutoff = max(0, min(xs.size, cutoff))
                line_cmd.set_data(xs[:cutoff], ys[:cutoff])
                line_cmd.set_3d_properties(zs[:cutoff])
        else:
            line_cmd.set_data([], [])
            line_cmd.set_3d_properties([])

    def _refresh_plan_display() -> None:
        nonlocal plan_lines, plan_arrows
        for line in plan_lines:
            line.remove()
        for arrow_group in plan_arrows:
            for arrow in arrow_group:
                arrow.remove()
        plan_lines = []
        plan_arrows = []

        entries = run_state.get("plan_entries")
        if not isinstance(entries, list) or not entries:
            return

        finite_entries = [
            entry
            for entry in entries
            if isinstance(entry, dict) and np.isfinite(entry.get("time", float("nan")))
        ]
        if not finite_entries:
            return

        finite_entries.sort(key=lambda entry: float(entry["time"]))
        times = np.array([float(entry["time"]) for entry in finite_entries], dtype=float)
        current_time = _current_time()
        if current_time is None:
            center_idx = 0
        else:
            center_idx = int(np.searchsorted(times, current_time, side="right"))
            if center_idx >= len(times):
                center_idx = len(times) - 1
            elif center_idx > 0:
                prev_time = times[center_idx - 1]
                next_time = times[center_idx] if center_idx < len(times) else times[-1]
                if abs(current_time - prev_time) <= abs(next_time - current_time):
                    center_idx -= 1

        start = max(0, center_idx - 5)
        end = min(len(finite_entries), center_idx + 5 + 1)
        for entry in finite_entries[start:end]:
            xs = np.asarray(entry["x"], dtype=float)
            ys = np.asarray(entry["y"], dtype=float)
            zs = np.asarray(entry["z"], dtype=float)
            if xs.size == 0 or ys.size == 0 or zs.size == 0:
                continue
            line = ax.plot(
                xs,
                ys,
                zs,
                color="tab:green",
                alpha=0.8,
                linewidth=1.4,
            )[0]
            plan_lines.append(line)
            arrows = _draw_plan_arrow(ax, xs, ys, zs)
            if arrows:
                plan_arrows.append(arrows)

    def update_position_marker() -> None:
        _update_control_history()
        _refresh_plan_display()

        odom_coords = run_state.get("odom_coords")
        title_text = run_state.get("title", "")
        count = int(run_state.get("position_count", 0))
        if not isinstance(odom_coords, tuple) or len(odom_coords) != 3 or count == 0:
            point_marker._offsets3d = ([], [], [])
            title.set_text(title_text)
            return
        xs, ys, zs = odom_coords
        index = int(run_state.get("position_index", 0))
        index %= count
        run_state["position_index"] = index
        point_marker._offsets3d = (
            np.asarray([xs[index]], dtype=float),
            np.asarray([ys[index]], dtype=float),
            np.asarray([zs[index]], dtype=float),
        )
        current_time = _current_time()
        if current_time is not None:
            title.set_text(f"{title_text} (t={current_time:.2f}s)")
        else:
            title.set_text(title_text)
        fig.canvas.draw_idle()

    def on_next(event: Optional[object]) -> None:
        if not run_state["runs"]:
            return
        run_state["index"] = (run_state["index"] + 1) % len(run_state["runs"])
        update_plot()

    def on_prev(event: Optional[object]) -> None:
        if not run_state["runs"]:
            return
        run_state["index"] = (run_state["index"] - 1) % len(run_state["runs"])
        update_plot()

    def on_slider_change(val: float) -> None:
        if run_state.get("suppress_slider_callback"):
            return
        count = int(run_state.get("position_count", 0))
        if count <= 0:
            run_state["position_index"] = 0
        elif count == 1:
            run_state["position_index"] = 0
        else:
            index = int(round(val * (count - 1)))
            index = max(0, min(count - 1, index))
            run_state["position_index"] = index
        update_position_marker()
        slider = run_state.get("slider")
        if isinstance(slider, Slider):
            slider.valtext.set_text(str(run_state["position_index"]))

    def _update_slider_state() -> None:
        slider = run_state.get("slider")
        if not isinstance(slider, Slider):
            return

        count = int(run_state.get("position_count", 0))
        index = int(run_state.get("position_index", 0))
        if count <= 0:
            count = 0
            index = 0
        elif count == 1:
            index = 0
        else:
            index = max(0, min(count - 1, index))
        run_state["position_index"] = index

        run_state["suppress_slider_callback"] = True
        try:
            if count <= 1:
                slider.set_active(False)
                slider.set_val(0.0)
                slider.valtext.set_text("0")
            else:
                slider.set_active(True)
                slider_value = index / (count - 1)
                slider.set_val(slider_value)
                slider.valtext.set_text(str(index))
        finally:
            run_state["suppress_slider_callback"] = False

    # Button layout
    btn_ax_prev = fig.add_axes([0.18, 0.08, 0.18, 0.05])
    btn_ax_next = fig.add_axes([0.64, 0.08, 0.18, 0.05])
    slider_ax = fig.add_axes([0.20, 0.02, 0.60, 0.03])
    run_state["slider_ax"] = slider_ax

    btn_prev = Button(btn_ax_prev, "Run ◀")
    btn_next = Button(btn_ax_next, "Run ▶")

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    slider = Slider(
        slider_ax,
        "Sample",
        0.0,
        1.0,
        valinit=0.0,
        valstep=None,
    )
    slider.on_changed(on_slider_change)
    run_state["slider"] = slider

    # Initial plot
    update_plot()
    plt.subplots_adjust(bottom=0.22)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot path following runs from parsed control/plan CSVs."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the path_follow_{GUID} outputs (control and plan CSVs).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    directory = args.input_dir
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")

    control_df, plan_df = load_dataframes(directory)
    runs = get_run_sequence(control_df)
    plot_tracking_summary(control_df)
    plot_runs(control_df, plan_df, runs)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
