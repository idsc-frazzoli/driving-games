from dataclasses import dataclass
from decimal import Decimal
from itertools import product
from typing import List, Tuple, Callable, Set

import numpy as np

from dg_commons import logger, Timestamp
from dg_commons.planning.trajectory import Trajectory
from dg_commons.time import time_function
from dg_commons.types import LinSpaceTuple
from sim.models.vehicle import VehicleState, VehicleCommands
from sim.models.vehicle_utils import VehicleParameters


@dataclass
class MPGParam:
    dt: Decimal
    n_steps: int
    velocity: LinSpaceTuple
    steering: LinSpaceTuple

    def __post_init__(self):
        assert isinstance(self.dt, Decimal)

    @classmethod
    def from_vehicle_parameters(cls, dt: Decimal, n_steps: int, n_vel: int, n_steer: int,
                                vp: VehicleParameters) -> "MPGParam":
        """
        :param dt:
        :param n_steps:
        :param n_vel:
        :param n_steer:
        :param vp:
        :return:
        """
        vel_linspace = (vp.vx_limits[0], vp.vx_limits[1], n_vel)
        steer_linspace = (-vp.delta_max, vp.delta_max, n_steer)
        return MPGParam(dt=dt,
                        n_steps=n_steps,
                        velocity=vel_linspace,
                        steering=steer_linspace)


class MotionPrimitivesGenerator:
    def __init__(self,
                 mpg_param: MPGParam,
                 vehicle_dynamics: Callable[[VehicleState, VehicleCommands, Timestamp], VehicleState],
                 vehicle_params: VehicleParameters):
        self.param = mpg_param
        self.vehicle_dynamics = vehicle_dynamics
        self.vehicle_params = vehicle_params

    @time_function
    def generate_motion_primitives(self, ) -> Set[Trajectory]:
        v_samples, steer_samples = self.generate_samples()
        logger.info(f"Attempting to generate {(len(v_samples) * len(steer_samples)) ** 2} motion primitives")
        motion_primitives: Set[Trajectory] = set()
        for (v_start, sa_start) in product(v_samples, steer_samples):
            for (v_end, sa_end) in product(v_samples, steer_samples):
                is_valid, input_a, input_sa_rate = self.check_input_constraints(
                    v_start, v_end, sa_start, sa_end)
                if not is_valid:
                    continue
                init_state = VehicleState(x=0, y=0, theta=0, vx=v_start, delta=sa_start)
                timestamps = [Decimal(0), ]
                states = [init_state, ]
                next_state = init_state
                cmds = VehicleCommands(acc=input_a, ddelta=input_sa_rate)
                for n_step in range(1, self.param.n_steps + 1):
                    next_state = self.vehicle_dynamics(next_state, cmds, float(self.param.dt))
                    timestamps.append(n_step * self.param.dt)
                    states.append(next_state)
                motion_primitives.add(Trajectory(timestamps=timestamps, values=states))
        logger.info(f"Found {len(motion_primitives)} feasible motion primitives")
        return motion_primitives

    def generate_samples(self) -> (List, List):
        # fixme list or numpy
        v_samples = np.linspace(*self.param.velocity)
        steer_samples = np.linspace(*self.param.steering)
        return v_samples, steer_samples

    def check_input_constraints(self, v_start, v_end, sa_start, sa_end) -> Tuple[bool, float, float]:
        """
        :param v_start: initial velocity
        :param v_end: ending velocity
        :param sa_start: initial steering angle
        :param sa_end: ending steering angle
        :return: [is_valid,acc,steer_rate]
        """
        horizon = float(self.param.dt * self.param.n_steps)
        acc = (v_end - v_start) / horizon
        sa_rate = (sa_end - sa_start) / horizon
        if not (-self.vehicle_params.ddelta_max <= sa_rate <= self.vehicle_params.ddelta_max) or \
                (not self.vehicle_params.acc_limits[0] <= acc <= self.vehicle_params.acc_limits[1]):
            return False, 0, 0
        else:
            return True, acc, sa_rate
