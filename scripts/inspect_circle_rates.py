from __future__ import annotations

import numpy as np
from pathlib import Path

log = Path("logs/path_follower_circle.csv")
data = np.genfromtxt(log, delimiter=",", names=True)
indices = [0, len(data)//4, len(data)//2, 3*len(data)//4, -1]
for idx in indices:
    row = data[idx]
    print(
        f"t={row['time_s']:.2f}s cmd_rate=({row['body_rate_command_x']:.2f}, {row['body_rate_command_y']:.2f}, {row['body_rate_command_z']:.2f})"
        f" actual_rate=({row['veh_body_rate_x']:.2f}, {row['veh_body_rate_y']:.2f}, {row['veh_body_rate_z']:.2f})"
    )
