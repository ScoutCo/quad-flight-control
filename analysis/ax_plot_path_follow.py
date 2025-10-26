import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


PLAN_LOOKAHEAD_SEGMENTS = 1
COMMAND_HISTORY_STRIDE = 15
COMMAND_VELOCITY_STRIDE = 15

COMMAND_HISTORY_COLOR = "#f28e1c"
CURRENT_COMMAND_COLOR = "#d55e00"
COMMAND_VELOCITY_HISTORY_COLOR = "#c4b5fd"
CURRENT_COMMAND_VELOCITY_COLOR = "#4c1d95"
PLAN_HISTORY_COLOR = "#2ca02c"
PLAN_CURRENT_COLOR = "#0c5b2a"
PLAN_LOOKAHEAD_COLOR = "#8fd18f"
PLAN_HISTORY_LINEWIDTH = 0.8
PLAN_CURRENT_LINEWIDTH = 2.0
PLAN_LOOKAHEAD_LINEWIDTH = 1.0
PLAN_LOOKAHEAD_LINESTYLE = ":"


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


def _yaw_from_quaternion(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray
) -> np.ndarray:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def _draw_plan_arrow(
    ax,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    *,
    color: str = PLAN_HISTORY_COLOR,
    alpha: float = 0.7,
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
            color=color,
            alpha=alpha,
            arrow_length_ratio=0.35,
            linewidth=1.0,
        )
        arrows.append(arrow)
    return arrows


def _extract_log_id(directory: Path) -> str:
    name = directory.name
    if name:
        cleaned = name.removeprefix("path_follow_")
        if cleaned:
            return cleaned
    resolved_name = directory.resolve().name
    return resolved_name.removeprefix("path_follow_")


def plot_tracking_summary(control_df: pd.DataFrame, log_id: str = "") -> plt.Figure | None:
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
        raise KeyError(
            f"Missing expected columns in control dataframe: {missing_columns}"
        )

    time = _extract_time_axis(control_df)
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1.2, 1, 0.6]},
    )
    ax_pos, ax_vel, ax_yaw, ax_run = axes
    title = "Control vs Odometry Tracking"
    fig.suptitle(title)
    if log_id:
        fig.subplots_adjust(top=0.88)
        fig.text(0.5, 0.93, f"Log: {log_id}", ha="center", fontsize="medium")
    ax_pos.set_title("Position Tracking")

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

    for component in ("x", "y", "z"):
        ax_vel.plot(
            time,
            control_df[f"odom_vel_{component}"].to_numpy(dtype=float, copy=False),
            label=f"odom_{component}",
            color=colors[component],
            linestyle="-",
        )
        ax_vel.plot(
            time,
            control_df[f"cmd_vel_{component}"].to_numpy(dtype=float, copy=False),
            label=f"cmd_{component}",
            color=colors[component],
            linestyle="--",
        )

    ax_vel.set_ylabel("Velocity (m/s)")
    ax_vel.legend(loc="upper right", ncol=3, fontsize="small")
    ax_vel.set_title("Velocity Tracking")
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
    log_id: str = "",
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
    if log_id:
        fig.subplots_adjust(top=0.90)
        fig.text(0.5, 0.94, f"Log: {log_id}", ha="center", fontsize="medium")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    line_odom = ax.plot([], [], [], lw=2, color="tab:blue", label="odom")[0]
    command_history = Line3DCollection(
        np.empty((0, 2, 3), dtype=float),
        colors=(COMMAND_HISTORY_COLOR,),
        linewidths=(0.8,),
        alpha=0.7,
    )
    ax.add_collection3d(command_history)
    command_history.set_visible(False)
    current_command_vector = ax.plot(
        [], [], [], lw=2.0, color=CURRENT_COMMAND_COLOR
    )[0]
    current_command_vector.set_visible(False)
    point_marker = ax.scatter(
        [],
        [],
        [],
        s=60,
        color="tab:blue",
        edgecolor="k",
        linewidth=0.5,
        alpha=0.9,
        depthshade=False,
        label="current position",
    )
    plan_history_proxy = Line2D(
        [0],
        [0],
        color=PLAN_HISTORY_COLOR,
        linewidth=PLAN_HISTORY_LINEWIDTH,
        alpha=0.65,
        label="plan history",
    )
    plan_current_proxy = Line2D(
        [0],
        [0],
        color=PLAN_CURRENT_COLOR,
        linewidth=PLAN_CURRENT_LINEWIDTH,
        alpha=0.9,
        label="plan current",
    )
    plan_lookahead_proxy = Line2D(
        [0],
        [0],
        color=PLAN_LOOKAHEAD_COLOR,
        linewidth=PLAN_LOOKAHEAD_LINEWIDTH,
        linestyle=PLAN_LOOKAHEAD_LINESTYLE,
        alpha=0.8,
        label="plan lookahead",
    )
    cmd_current_proxy = Line2D(
        [0],
        [0],
        color=CURRENT_COMMAND_COLOR,
        linewidth=2.0,
        label="cmd position",
    )
    cmd_history_proxy = Line2D(
        [0],
        [0],
        color=COMMAND_HISTORY_COLOR,
        linewidth=0.8,
        alpha=0.7,
        label="cmd position history",
    )
    vel_current_proxy = Line2D(
        [0],
        [0],
        color=CURRENT_COMMAND_VELOCITY_COLOR,
        linewidth=2.0,
        label="cmd velocity",
    )
    vel_history_proxy = Line2D(
        [0],
        [0],
        color=COMMAND_VELOCITY_HISTORY_COLOR,
        linewidth=0.8,
        alpha=0.7,
        label="cmd velocity history",
    )
    plan_handles = [plan_history_proxy, plan_current_proxy, plan_lookahead_proxy]
    velocity_handles = [vel_current_proxy, vel_history_proxy]
    legend_handles = [line_odom, cmd_current_proxy, cmd_history_proxy, point_marker]
    legend_handles.extend(velocity_handles)
    if plan_available:
        legend_handles.extend(plan_handles)
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
        "cmd_vel_coords": (np.array([], dtype=float),) * 3,
        "velocity_history_quivers": [],
        "current_velocity_quiver": None,
        "plan_segments": [],
        "plan_sorted_indices": [],
        "plan_always_visible": [],
        "plan_visible_indices": set(),
        "plan_needs_refresh": False,
        "title": "All samples",
        "position_count": 0,
        "slider": None,
        "suppress_slider_callback": False,
    }
    plan_segments: list[dict[str, object]] = []

    def update_plot(run_idx: Optional[int] = None) -> None:
        nonlocal plan_segments
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

        # Clear previous velocity arrows
        existing_velocity_quivers = run_state.get("velocity_history_quivers", [])
        if isinstance(existing_velocity_quivers, list):
            for quiver_artist in existing_velocity_quivers:
                if hasattr(quiver_artist, "remove"):
                    quiver_artist.remove()
        run_state["velocity_history_quivers"] = []
        current_velocity_quiver = run_state.get("current_velocity_quiver")
        if current_velocity_quiver is not None and hasattr(current_velocity_quiver, "remove"):
            current_velocity_quiver.remove()
        run_state["current_velocity_quiver"] = None

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
        if {
            "cmd_vel_x",
            "cmd_vel_y",
            "cmd_vel_z",
        }.issubset(run_slice.columns):
            vel_x = run_slice["cmd_vel_x"].to_numpy(dtype=float, copy=False)
            vel_y = run_slice["cmd_vel_y"].to_numpy(dtype=float, copy=False)
            vel_z = run_slice["cmd_vel_z"].to_numpy(dtype=float, copy=False)
        else:
            vel_x = vel_y = vel_z = np.array([], dtype=float)

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
            if vel_x.size:
                vel_x_valid = vel_x[cmd_indices]
                vel_y_valid = vel_y[cmd_indices]
                vel_z_valid = vel_z[cmd_indices]
            else:
                vel_x_valid = vel_y_valid = vel_z_valid = np.array([], dtype=float)
        else:
            cmd_x_valid = cmd_y_valid = cmd_z_valid = np.array([], dtype=float)
            vel_x_valid = vel_y_valid = vel_z_valid = np.array([], dtype=float)

        # Clear previous plan artists
        for segment in plan_segments:
            line_artist = segment.get("line")
            if hasattr(line_artist, "remove"):
                line_artist.remove()
            for arrow_artist in segment.get("arrows", []):
                if hasattr(arrow_artist, "remove"):
                    arrow_artist.remove()
        plan_segments = []

        plan_bounds: list[np.ndarray] = []
        new_plan_segments: list[dict[str, object]] = []
        if plan_slice is not None and not plan_slice.empty:
            plan_times = plan_slice.index.to_numpy(dtype=float, copy=False)
            plan_x_cols = [f"step{idx}_x" for idx in step_indices]
            plan_y_cols = [f"step{idx}_y" for idx in step_indices]
            plan_z_cols = [f"step{idx}_z" for idx in step_indices]
            plan_x_values = plan_slice[plan_x_cols].to_numpy(dtype=float, copy=False)
            plan_y_values = plan_slice[plan_y_cols].to_numpy(dtype=float, copy=False)
            plan_z_values = plan_slice[plan_z_cols].to_numpy(dtype=float, copy=False)
            run_slice_positions = run_slice[["odom_pos_x", "odom_pos_y", "odom_pos_z"]]
            if isinstance(run_slice_positions.index, pd.Index):
                odom_for_plan = run_slice_positions.reindex(
                    plan_slice.index, method="pad", tolerance=None
                )
            else:
                odom_for_plan = run_slice_positions

            for idx_row, (px, py, pz) in enumerate(
                zip(plan_x_values, plan_y_values, plan_z_values)
            ):
                valid = np.isfinite(px) & np.isfinite(py) & np.isfinite(pz)
                if not np.any(valid):
                    continue
                px_valid = px[valid]
                py_valid = py[valid]
                pz_valid = pz[valid]
                entry_time = (
                    float(plan_times[idx_row])
                    if idx_row < plan_times.size
                    else float("nan")
                )
                origin_point = odom_for_plan.iloc[idx_row].to_numpy(dtype=float, copy=True)
                if not np.all(np.isfinite(origin_point)):
                    continue
                combined_x = np.concatenate(([origin_point[0]], px_valid))
                combined_y = np.concatenate(([origin_point[1]], py_valid))
                combined_z = np.concatenate(([origin_point[2]], pz_valid))
                plan_bounds.append(np.column_stack((combined_x, combined_y, combined_z)))
                line = ax.plot(
                    combined_x,
                    combined_y,
                    combined_z,
                    color=PLAN_HISTORY_COLOR,
                    alpha=0.65,
                    linewidth=PLAN_HISTORY_LINEWIDTH,
                )[0]
                line.set_visible(False)
                arrows_raw = _draw_plan_arrow(
                    ax,
                    combined_x,
                    combined_y,
                    combined_z,
                    color=PLAN_HISTORY_COLOR,
                    alpha=0.65,
                )
                arrow_list = list(arrows_raw) if arrows_raw else []
                for arrow_artist in arrow_list:
                    if hasattr(arrow_artist, "set_visible"):
                        arrow_artist.set_visible(False)
                new_plan_segments.append(
                    {
                        "time": entry_time,
                        "line": line,
                        "arrows": arrow_list,
                    }
                )

        plan_segments = new_plan_segments

        run_state["plan_segments"] = plan_segments
        finite_indices = [
            idx
            for idx, segment in enumerate(plan_segments)
            if np.isfinite(float(segment.get("time", float("nan"))))
        ]
        finite_indices.sort(key=lambda idx: float(plan_segments[idx]["time"]))
        always_visible = [
            idx
            for idx, segment in enumerate(plan_segments)
            if not np.isfinite(float(segment.get("time", float("nan"))))
        ]
        run_state["plan_sorted_indices"] = finite_indices
        run_state["plan_always_visible"] = always_visible
        run_state["plan_visible_indices"] = set()
        run_state["odom_coords"] = (odom_x_valid, odom_y_valid, odom_z_valid)
        run_state["cmd_coords"] = (cmd_x_valid, cmd_y_valid, cmd_z_valid)
        run_state["cmd_vel_coords"] = (vel_x_valid, vel_y_valid, vel_z_valid)
        run_state["position_count"] = int(odom_indices.size)
        time_values = _extract_time_axis(run_slice)
        run_state["odom_times"] = (
            time_values[odom_indices]
            if odom_indices.size
            else np.array([], dtype=float)
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
            bounds_components.append(
                np.column_stack((cmd_x_valid, cmd_y_valid, cmd_z_valid))
            )
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
        ax.set_zlim(
            upper[2], lower[2]
        )  # In NED frame, smaller (more negative) values are higher altitude
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
        idx = int(run_state.get("position_index", 0))
        coords = run_state.get("odom_coords")
        xs_vehicle = np.array([], dtype=float)
        ys_vehicle = np.array([], dtype=float)
        zs_vehicle = np.array([], dtype=float)
        if (
            isinstance(coords, tuple)
            and len(coords) == 3
            and all(isinstance(arr, np.ndarray) for arr in coords)
        ):
            xs_vehicle, ys_vehicle, zs_vehicle = coords
            upto = min(idx + 1, xs_vehicle.size)
            line_odom.set_data(xs_vehicle[:upto], ys_vehicle[:upto])
            line_odom.set_3d_properties(zs_vehicle[:upto])
        else:
            line_odom.set_data([], [])
            line_odom.set_3d_properties([])

        existing_velocity_quivers = run_state.get("velocity_history_quivers", [])
        if isinstance(existing_velocity_quivers, list):
            for quiver_artist in existing_velocity_quivers:
                if hasattr(quiver_artist, "remove"):
                    quiver_artist.remove()
        run_state["velocity_history_quivers"] = []
        current_velocity_quiver = run_state.get("current_velocity_quiver")
        if current_velocity_quiver is not None and hasattr(current_velocity_quiver, "remove"):
            current_velocity_quiver.remove()
        run_state["current_velocity_quiver"] = None

        cmd_coords = run_state.get("cmd_coords")
        cmd_times = run_state.get("cmd_times")
        current_time = _current_time()
        odom_times = run_state.get("odom_times")
        if not (
            isinstance(cmd_coords, tuple)
            and len(cmd_coords) == 3
            and all(isinstance(arr, np.ndarray) for arr in cmd_coords)
            and isinstance(cmd_times, np.ndarray)
            and isinstance(xs_vehicle, np.ndarray)
        ):
            command_history.set_segments([])
            command_history.set_visible(False)
            current_command_vector.set_data([], [])
            current_command_vector.set_3d_properties([])
            current_command_vector.set_visible(False)
            return

        xs_cmd, ys_cmd, zs_cmd = cmd_coords
        if xs_cmd.size == 0 or xs_vehicle.size == 0:
            command_history.set_segments([])
            command_history.set_visible(False)
            current_command_vector.set_data([], [])
            current_command_vector.set_3d_properties([])
            current_command_vector.set_visible(False)
            return

        if current_time is not None and np.isfinite(current_time):
            cutoff = int(np.searchsorted(cmd_times, current_time + 1e-9, side="right"))
        else:
            cutoff = min(idx + 1, xs_cmd.size)
        cutoff = max(0, min(xs_cmd.size, cutoff))
        if cutoff <= 0:
            command_history.set_segments([])
            command_history.set_visible(False)
            current_command_vector.set_data([], [])
            current_command_vector.set_3d_properties([])
            current_command_vector.set_visible(False)
            return

        finite_mask = (
            np.isfinite(xs_cmd[:cutoff])
            & np.isfinite(ys_cmd[:cutoff])
            & np.isfinite(zs_cmd[:cutoff])
        )
        finite_mask &= np.isfinite(cmd_times[:cutoff])
        valid_indices = np.nonzero(finite_mask)[0]
        if valid_indices.size == 0:
            command_history.set_segments([])
            command_history.set_visible(False)
            current_command_vector.set_data([], [])
            current_command_vector.set_3d_properties([])
            current_command_vector.set_visible(False)
            return

        cmd_times_valid = cmd_times[valid_indices]
        if isinstance(odom_times, np.ndarray) and odom_times.size:
            vehicle_indices = np.searchsorted(odom_times, cmd_times_valid, side="right") - 1
            vehicle_indices = np.clip(vehicle_indices, 0, odom_times.size - 1)
        else:
            vehicle_indices = np.clip(valid_indices, 0, xs_vehicle.size - 1)

        vehicle_points = np.column_stack(
            (
                xs_vehicle[vehicle_indices],
                ys_vehicle[vehicle_indices],
                zs_vehicle[vehicle_indices],
            )
        )
        command_points = np.column_stack(
            (
                xs_cmd[valid_indices],
                ys_cmd[valid_indices],
                zs_cmd[valid_indices],
            )
        )
        command_points_full = command_points.copy()
        paired_segments = np.stack((vehicle_points, command_points), axis=1)
        stride = max(1, int(COMMAND_HISTORY_STRIDE))
        if stride > 1 and paired_segments.shape[0] > 1:
            selector = np.arange(paired_segments.shape[0], dtype=int)[::stride]
            if selector[-1] != paired_segments.shape[0] - 1:
                selector = np.append(selector, paired_segments.shape[0] - 1)
            paired_segments = paired_segments[selector]

        command_history.set_segments(paired_segments)
        command_history.set_visible(True)

        last_segment = paired_segments[-1]
        current_command_vector.set_data(
            [last_segment[0, 0], last_segment[1, 0]],
            [last_segment[0, 1], last_segment[1, 1]],
        )
        current_command_vector.set_3d_properties(
            [last_segment[0, 2], last_segment[1, 2]]
        )
        current_command_vector.set_visible(True)

        velocity_coords = run_state.get("cmd_vel_coords")
        if not (
            isinstance(velocity_coords, tuple)
            and len(velocity_coords) == 3
            and all(isinstance(arr, np.ndarray) for arr in velocity_coords)
        ):
            return

        vx_all, vy_all, vz_all = velocity_coords
        if vx_all.size == 0 or vy_all.size == 0 or vz_all.size == 0:
            return

        vx_subset = vx_all[:cutoff]
        vy_subset = vy_all[:cutoff]
        vz_subset = vz_all[:cutoff]
        if vx_subset.size == 0:
            return

        velocity_mask = (
            finite_mask
            & np.isfinite(vx_subset)
            & np.isfinite(vy_subset)
            & np.isfinite(vz_subset)
        )
        velocity_indices = np.nonzero(velocity_mask)[0]
        if velocity_indices.size == 0:
            return

        shared_indices = np.intersect1d(
            valid_indices, velocity_indices, assume_unique=True
        )
        if shared_indices.size == 0:
            return

        mapping = np.searchsorted(valid_indices, shared_indices)
        starts = command_points_full[mapping]
        velocity_vectors = np.column_stack(
            (
                vx_subset[shared_indices],
                vy_subset[shared_indices],
                vz_subset[shared_indices],
            )
        )

        history_stride = max(1, int(COMMAND_VELOCITY_STRIDE))
        indices_order = np.arange(shared_indices.size, dtype=int)
        history_selector = indices_order[::history_stride]
        if history_selector.size and history_selector[-1] == indices_order[-1]:
            history_selector = history_selector[:-1]

        history_quivers: list = []
        for hist_idx in history_selector:
            start = starts[hist_idx]
            vector = velocity_vectors[hist_idx]
            if not np.all(np.isfinite(vector)):
                continue
            quiver_artist = ax.quiver(
                start[0],
                start[1],
                start[2],
                vector[0],
                vector[1],
                vector[2],
                color=COMMAND_VELOCITY_HISTORY_COLOR,
                linewidth=0.8,
                arrow_length_ratio=0.25,
                alpha=0.7,
            )
            history_quivers.append(quiver_artist)

        run_state["velocity_history_quivers"] = history_quivers

        start_current = starts[-1]
        vector_current = velocity_vectors[-1]
        if np.all(np.isfinite(vector_current)):
            current_quiver_artist = ax.quiver(
                start_current[0],
                start_current[1],
                start_current[2],
                vector_current[0],
                vector_current[1],
                vector_current[2],
                color=CURRENT_COMMAND_VELOCITY_COLOR,
                linewidth=2.0,
                arrow_length_ratio=0.25,
            )
            run_state["current_velocity_quiver"] = current_quiver_artist

    def _refresh_plan_display() -> None:
        segments = run_state.get("plan_segments")
        if not isinstance(segments, list) or not segments:
            return

        def _apply_plan_style(line_artist, arrows, category: str) -> None:
            if not hasattr(line_artist, "set_color"):
                return
            if category == "current":
                color = PLAN_CURRENT_COLOR
                linewidth = PLAN_CURRENT_LINEWIDTH
                linestyle = "-"
                alpha = 0.9
            elif category == "lookahead":
                color = PLAN_LOOKAHEAD_COLOR
                linewidth = PLAN_LOOKAHEAD_LINEWIDTH
                linestyle = PLAN_LOOKAHEAD_LINESTYLE
                alpha = 0.75
            else:
                color = PLAN_HISTORY_COLOR
                linewidth = PLAN_HISTORY_LINEWIDTH
                linestyle = "-"
                alpha = 0.65

            line_artist.set_color(color)
            line_artist.set_linewidth(linewidth)
            line_artist.set_linestyle(linestyle)
            if hasattr(line_artist, "set_alpha"):
                line_artist.set_alpha(alpha)

            for arrow_artist in arrows:
                if hasattr(arrow_artist, "set_color"):
                    arrow_artist.set_color(color)
                if hasattr(arrow_artist, "set_alpha"):
                    arrow_artist.set_alpha(alpha)

        current_time = _current_time()
        sorted_indices = run_state.get("plan_sorted_indices", [])
        always_visible = run_state.get("plan_always_visible", [])

        desired_visible: set[int] = set(always_visible)
        history_indices: set[int] = set(always_visible)
        lookahead_indices: set[int] = set()
        current_index: Optional[int] = None

        if not sorted_indices:
            history_indices.update(range(len(segments)))
            desired_visible.update(history_indices)
        else:
            sorted_times = np.array(
                [
                    float(segments[idx].get("time", float("nan")))
                    for idx in sorted_indices
                ],
                dtype=float,
            )
            if current_time is None or not np.isfinite(current_time):
                history_count = len(sorted_indices)
            else:
                history_count = int(
                    np.searchsorted(sorted_times, current_time + 1e-9, side="right")
                )
                history_count = max(0, min(len(sorted_indices), history_count))

            if history_count > 0:
                history_indices.update(sorted_indices[: history_count - 1])
                current_index = sorted_indices[history_count - 1]
                desired_visible.update(sorted_indices[:history_count])
            else:
                current_index = None

            if PLAN_LOOKAHEAD_SEGMENTS > 0:
                lookahead_end = min(
                    len(sorted_indices), history_count + PLAN_LOOKAHEAD_SEGMENTS
                )
                lookahead_slice = sorted_indices[history_count:lookahead_end]
                lookahead_indices.update(lookahead_slice)
                desired_visible.update(lookahead_slice)

        run_state["plan_visible_indices"] = desired_visible

        for idx, segment in enumerate(segments):
            visible = idx in desired_visible
            line_artist = segment.get("line")
            arrows = segment.get("arrows", [])
            if hasattr(line_artist, "set_visible"):
                line_artist.set_visible(visible)
            for arrow_artist in arrows:
                if hasattr(arrow_artist, "set_visible"):
                    arrow_artist.set_visible(visible)
            if not visible:
                continue

            if idx == current_index:
                _apply_plan_style(line_artist, arrows, "current")
            elif idx in lookahead_indices:
                _apply_plan_style(line_artist, arrows, "lookahead")
            else:
                _apply_plan_style(line_artist, arrows, "history")

    def update_position_marker(refresh_plan: bool = True) -> None:
        _update_control_history()
        if refresh_plan:
            _refresh_plan_display()
            run_state["plan_needs_refresh"] = False
        else:
            run_state["plan_needs_refresh"] = True

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
        update_position_marker(refresh_plan=False)
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

    def on_mouse_release(event) -> None:
        if event.inaxes is slider_ax and run_state.get("plan_needs_refresh"):
            _refresh_plan_display()
            run_state["plan_needs_refresh"] = False
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_release_event", on_mouse_release)

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
    log_id = _extract_log_id(directory)
    plot_tracking_summary(control_df, log_id=log_id)
    plot_runs(control_df, plan_df, runs, log_id=log_id)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
