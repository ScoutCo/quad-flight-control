# Quadcopter Controls and Simulation Tools

## Project Layout
- `quad_flight_control`: main package
  - `sim/`: simple kinematic simulator to generate represntative poses from position + velocity controls. no dynamics
  - `path_follower/`: setpoint control generation from a plan; port of C++ code
  - `common/`: reusable bits for sim and flight
- `analysis/`: notebooks and ad-hoc studies
- `examples/`: runnable demos that tie the simulator, command generator, and trajectories together
- `flight_test/`: MAVSDK-based scripts for exercising the follower against a real vehicle.
- `plots/`: viz and plotting tools
- `tests/`: unit tests 

## Quick Start
Run one of the bundled demos to generate telemetry while following a reference trajectory:

```bash
uv run python -m examples.path_follower_circle
```

The script will print a brief analysis summary and write a CSV log under `logs/` that you can inspect in your plotting tool of choice.

## Development
Install the project (and optional development dependencies) with uv:

```bash
uv sync --all-extras --dev
```

Execute the simulator tests with:

```bash
pytest tests
```

Telemetry logs and intermediate artefacts are written to the `logs/` directory.

## Minimal Usage Example
```python
from path_follower import PositionVelocityPathFollower
from sim import Simulator

follower = PositionVelocityPathFollower()
simulator = Simulator()

while True:
    new_plan = try_get_new_plan()
    cmd = follower.next_command(t).command
    step_data = flight_sim.step(cmd)

    pos = step_data.state.position_ned
    ...

    sleep(dt)
```
