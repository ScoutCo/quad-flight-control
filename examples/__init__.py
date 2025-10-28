"""Lightweight example scripts for the simulator path follower demos."""

from quad_flight_control.common import (
    Trajectory,
    circle_trajectory,
    line_trajectory,
    sinusoid_trajectory,
    zigzag_trajectory,
)
from .common import run_path_follower_example

__all__ = [
    "Trajectory",
    "circle_trajectory",
    "line_trajectory",
    "sinusoid_trajectory",
    "zigzag_trajectory",
    "run_path_follower_example",
]
