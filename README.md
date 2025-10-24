# Multirotor Simulation Toolkit

This repository bundles a lightweight multirotor simulator together with supporting utilities for generating reference plans, running path-following controllers, and analysing the resulting telemetry. It focuses on a low-complexity fixed-step model that is easy to iterate on when evaluating offboard control strategies.

## What’s Included
- `sim/`: core simulator, state containers, telemetry logger, and common math helpers.
- `offboard_control/`: path follower implementation that operates on position/velocity plans.
- `examples/`: helper functions and runnable demos showcasing the simulator and path follower.
- `tests/sim/`: unit tests that exercise the simulator and logging utilities.

## Quick Start
Run one of the bundled demos to generate telemetry while following a reference trajectory:

```bash
python -m examples.path_follower_circle
```

The script will print a brief analysis summary and write a CSV log under `logs/` that you can inspect in your plotting tool of choice.

## Development
Install the project (and optional development dependencies) with uv:

```bash
uv sync --all-extras --dev
```

Execute the simulator tests with:

```bash
pytest tests/sim
```

Telemetry logs and intermediate artefacts are written to the `logs/` directory.
