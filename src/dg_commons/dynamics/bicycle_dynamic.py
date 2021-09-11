import math
from dataclasses import replace
from decimal import Decimal as D
from typing import Mapping, Tuple, List, Any, FrozenSet

import numpy as np

from dg_commons import Timestamp
from games import Dynamics, U, X, SR
from possibilities import Poss
from sim.models.vehicle import VehicleState, VehicleCommands
from sim.models.vehicle_structures import VehicleGeometry
from sim.models.vehicle_utils import VehicleParameters

__all__ = ["BicycleDynamics"]


class BicycleDynamics(Dynamics[VehicleState, VehicleCommands, Any]):

    def __init__(self, vg: VehicleGeometry, vp: VehicleParameters):
        self.vg: VehicleGeometry = vg
        self.vp: VehicleParameters = vp
        """ The vehicle parameters"""

    def all_actions(self) -> FrozenSet[U]:
        pass

    def successors(self, x: VehicleState, u0: VehicleCommands, dt: D = None) \
            -> Mapping[VehicleCommands, Poss[VehicleState]]:
        """ For each state, returns a dictionary U -> Possible Xs """
        # todo
        pass

    def successor(self, x0: VehicleState, u: VehicleCommands, dt: Timestamp) -> VehicleState:
        """ Perform Euler forward integration to propagate state using actions for time dt """
        dt = float(dt)
        # input constraints
        acc = float(np.clip(u.ddelta, self.vp.acc_limits[0], self.vp.acc_limits[1]))
        ddelta = float(np.clip(u.ddelta, - self.vp.ddelta_max, self.vp.ddelta_max))

        state_rate = self.dynamics(x0, replace(u, acc=acc, ddelta=ddelta))
        x0 += state_rate * dt

        # state constraints
        vx = float(np.clip(x0.vx, self.vp.acc_limits[0], self.vp.acc_limits[1]))
        delta = float(np.clip(x0.delta, -self.vp.ddelta_max, self.vp.ddelta_max))

        new_state = replace(x0, vx=vx, delta=delta)
        return new_state

    def successor_ivp(self, x0: VehicleState, u: VehicleCommands, dt: D, dt_samp: D) \
            -> Tuple[VehicleState, List[VehicleState]]:
        # todo
        pass

    def dynamics(self, x0: VehicleState, u: VehicleCommands) -> VehicleState:
        """ Get rate of change of states for given control inputs """

        dx = x0.vx
        dtheta = dx * math.tan(x0.delta) / self.vg.length
        dy = dtheta * self.vg.lr
        costh = math.cos(x0.theta)
        sinth = math.sin(x0.theta)
        xdot = dx * costh - dy * sinth
        ydot = dx * sinth + dy * costh
        x_rate = VehicleState(x=xdot, y=ydot, theta=dtheta, vx=u.acc, delta=u.ddelta)
        return x_rate

    def get_shared_resources(self, x: X) -> FrozenSet[SR]:
        pass
