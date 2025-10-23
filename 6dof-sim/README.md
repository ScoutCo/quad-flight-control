# 6DOF Flight Dynamics Simulator

This package provides a lightweight, fixed-time-step 6DOF simulator suitable for iterating on cascaded multirotor control logic. It mirrors the SET_POSITION_TARGET_LOCAL_NED interface by accepting position setpoints with velocity feedforward, and models a vehicle with configurable control loops, actuator lags, and simple environmental disturbances.

## Key Features
- Cascaded position → velocity → attitude → rate control structure with first-order lags and PID elements.
- Rigid-body dynamics with quaternion attitude, aerodynamic drag, and configurable external forces/moments.
- First-order actuator models with thrust and torque saturation.
- Deterministic fixed-step RK4 integrator capable of running faster than real time.
- Easily parameterized via dataclasses for vehicle, controller, actuator, and environment settings.

## Layout
```
6dof-sim/
├─ sixdof_sim/
│  ├─ __init__.py
│  ├─ actuators.py
│  ├─ commands.py
│  ├─ control.py
│  ├─ dynamics.py
│  ├─ environment.py
│  ├─ math_utils.py
│  ├─ parameters.py
│  ├─ simulator.py
│  └─ states.py
└─ examples/
   └─ basic_sim.py
```

## Quick Start
```bash
python 6dof-sim/examples/basic_sim.py
```
The script runs a 10 s hover command and prints the final state and actuator outputs. Adjust the command callback to inject different trajectories or wind disturbances via `EnvironmentParameters`.

## Next Steps
- Integrate onboard path-following controller & test against generated trajectories
- Derive vehicle parameters from actual hardware or ArduPilot logs.
- Add wind gust models or time-varying disturbance callbacks.
- Hook the simulator to live MAVLink setpoint streams for software-in-the-loop testing.
