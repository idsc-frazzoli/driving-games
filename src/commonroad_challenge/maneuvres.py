
import numpy as np
from commonroad.scenario.trajectory import Trajectory

from dg_commons import Timestamp
from dg_commons.sim.models.vehicle import VehicleState, VehicleCommands
from trajectory_games import TrajectoryGenParams, BicycleDynamics
from decimal import Decimal as D

# todo: why does this yield infeasible trajectories when delta of initial state is different from 0?
# -> because the friction circle is violated!

def emergency_braking_trajectory(state: VehicleState,
                                 max_long_acc: float,
                                 params: TrajectoryGenParams,
                                 t_final: Timestamp) -> Trajectory:
    bicycle_dyn = BicycleDynamics(params=params)
    dt = float(params.dt)

    tol = 1e-5  # floating point errors

    timestamps = np.arange(start=0.0, stop=t_final + dt, step=dt)
    values = [state]
    next_state = state
    for _ in range(len(timestamps) - 1):
        dec_to_0 = (0.0 - next_state.vx) / dt
        dec_max = min(max_long_acc, abs(dec_to_0))

        if dec_max < tol:
            dec_max = 0.0
        # apply max available braking and continue in same direction, keeping steering constant
        u_emergency = VehicleCommands(acc=-dec_max, ddelta=0.0)
        state_1 = bicycle_dyn.successor_ivp(x0=(0, next_state), u=u_emergency, dt=D(dt), dt_samp=params.dt_samp)
        next_state = state_1[0][1]
        # if next_state.vx == -0.0:
        #     next_state.vx = 0.0
        # print(state_1[0][1])
        # print(state_0)
        values.append(next_state)

    return Trajectory(timestamps=timestamps, values=values)


# todo: why is the value of theta exploding?
def braking_trajectory(state: VehicleState,
                       long_dec: float,
                       max_long_acc: float,
                       params: TrajectoryGenParams,
                       t_final: Timestamp) -> Trajectory:
    bicycle_dyn = BicycleDynamics(params=params)
    dt = float(params.dt)

    timestamps = np.arange(start=0.0, stop=t_final + dt, step=dt)
    values = [state]
    next_state = state
    for _ in range(len(timestamps) - 1):
        dec_max = min(max_long_acc, abs(long_dec))

        # apply max available braking and continue in same direction, keeping steering constant
        u_emergency = VehicleCommands(acc=1.0, ddelta=0.0)
        state_1 = bicycle_dyn.successor_ivp(x0=(0, next_state), u=u_emergency, dt=D(dt), dt_samp=params.dt_samp)
        next_state = state_1[0][1]
        # if next_state.vx == -0.0:
        #     next_state.vx = 0.0
        # print(state_1[0][1])
        # print(state_0)
        values.append(next_state)

    return Trajectory(timestamps=timestamps, values=values)