from __future__ import annotations

import pathlib
import sys

import numpy as np

THIS_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from sixdof_sim import PositionVelocityCommand, SixDofSimulator  # noqa: E402
from sixdof_sim.math_utils import quat_to_euler  # noqa: E402


def hover_command(_time: float, _state) -> PositionVelocityCommand:
    position = np.array([0.0, 0.0, -5.0])
    velocity_ff = np.zeros(3)
    return PositionVelocityCommand(position_ned=position, velocity_ned_ff=velocity_ff, yaw_heading=None)


def main() -> None:
    sim = SixDofSimulator(dt=0.01)
    history = sim.run(final_time=10.0, command_fn=hover_command)
    last = history[-1]
    euler = quat_to_euler(last.state.quaternion_bn)
    print("Final time", last.time)
    print("Position NED", last.state.position_ned)
    print("Velocity NED", last.state.velocity_ned)
    print("Euler (rad)", euler)
    print("Actuator thrust", last.actuators.thrust)


if __name__ == "__main__":
    main()
