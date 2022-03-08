import math
from decimal import Decimal as D
from itertools import product
from typing import FrozenSet, List, Mapping, Set, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from dg_commons import Timestamp
from .structures import TrajectoryParams
from dg_commons.sim.models.vehicle import VehicleState, VehicleCommands, VehicleGeometry

__all__ = ["BicycleDynamics"]


class BicycleDynamics:
    v_max: float
    v_min: float
    """ Maximum and Minimum velocities [m/s] """

    st_max: float
    """ Maximum steering angle [rad] """

    vg: VehicleGeometry
    """ The vehicle's geometry parameters"""

    u_acc: FrozenSet[float]
    """ Possible values of acceleration [m/s2] """

    u_dst: FrozenSet[float]
    """ Possible values of steering rate [rad/s] """

    def __init__(self, params: TrajectoryParams):
        self.v_max = params.v_max
        self.v_min = params.v_min
        self.st_max = params.st_max
        self.vg = params.vg
        self.u_acc = params.u_acc
        self.u_dst = params.u_dst
        if not self.u_acc:
            raise ValueError("No feasible acceleration")
        if not self.u_dst:
            raise ValueError("No feasible steering rate")

    def all_actions(self) -> Set[VehicleCommands]:
        res = set()
        for acc, dst in product(self.u_acc, self.u_dst):
            res.add(VehicleCommands(acc=acc, dst=dst))
        return res

    @staticmethod
    def get_clipped(val: float, lo: float, hi: float) -> float:
        """Get value clipped between limits"""
        if lo <= val <= hi:
            return val
        if val < lo:
            return lo
        return hi

    def get_feasible_acc(self, x: VehicleState, dt: D, u0: VehicleCommands) -> Set[float]:
        """Get feasible accelerations for current state with mean u0"""
        dt_f = float(dt)
        u_acc = set(
            [
                self.get_clipped(val=_ + u0.acc, lo=(self.v_min - x.vx) / dt_f, hi=(self.v_max - x.vx) / dt_f)
                for _ in self.u_acc
            ]
        )
        return u_acc

    def get_feasible_dst(self, x: VehicleState, dt: D, u0: VehicleCommands) -> Set[float]:
        """Get feasible steering rates for current state with mean u0"""
        dt_f = float(dt)
        u_dst = set(
            [
                self.get_clipped(
                    val=_ + u0.ddelta, lo=(-self.st_max - x.delta) / dt_f, hi=(self.st_max - x.delta)
                )
                for _ in self.u_dst
            ]
        )
        return u_dst

    def successors(self, x: VehicleState, dt: D, u0: VehicleCommands = None) -> Mapping[VehicleCommands, VehicleState]:
        """For each state, returns a dictionary U -> Possible Xs"""

        if u0 is None:
            u0 = VehicleCommands(acc=0.0, ddelta=0.0)
        u_acc = self.get_feasible_acc(x=x, dt=dt, u0=u0)
        u_dst = self.get_feasible_dst(x=x, dt=dt, u0=u0)

        res = {}
        for acc, dst in product(u_acc, u_dst):
            u = VehicleCommands(acc=acc, ddelta=dst)
            state, _ = self.successor_forward(x, u, dt)
            res[u] = state
        return res

    def successor(self, x0: VehicleState, u: VehicleCommands, dt: D) -> Tuple[VehicleState, List[VehicleState]]:
        """Perform RK2 integration to propagate state using actions for time dt"""
        dt_f = float(dt)
        vf = self.get_clipped(val=x0.vx + u.acc * dt_f, lo=self.v_min, hi=self.v_max)
        stf = self.get_clipped(val=x0.delta + u.ddelta * dt_f, lo=-self.st_max, hi=self.st_max)
        u_clip = VehicleCommands(acc=(vf - x0.vx) / dt_f, ddelta=(stf - x0.delta) / dt_f)

        k1 = self.dynamics(x0, u_clip)
        k2 = self.dynamics(x0 + k1 * dt_f, u_clip)
        ret = x0 + (k1 + k2) * (dt_f / 2.0)
        return ret, [x0, ret]

    def successor_forward(self, x0: VehicleState, u: VehicleCommands, dt: D) -> Tuple[VehicleState, List[VehicleState]]:
        """Perform Euler forward integration to propagate state using actions for time dt"""
        dt_f = float(dt)
        v0, st0 = x0.v, x0.st
        x0.v += u.acc * dt_f
        x0.st += u.dst * dt_f
        tol = 1e-3
        if not self.v_min - tol <= x0.v <= self.v_max + tol:
            print("Velocity outside limits")
        if not -self.st_max - tol <= x0.st <= self.st_max + tol:
            print(f"Steering = {x0.st}, outside limits")
        u = VehicleCommands(acc=0.0, dst=0.0)

        k1 = self.dynamics(x0, u)
        ret = x0 + k1 * dt_f
        x0.v, x0.st = v0, st0
        return ret, [x0, ret]

    def successor_ivp(
        self, x0: Tuple[Timestamp, VehicleState], u: VehicleCommands, dt: D, dt_samp: D
    ) -> Tuple[Tuple[Timestamp, VehicleState], List[Tuple[Timestamp, VehicleState]]]:
        """
        Perform initial value problem integration
        to propagate state using actions for time dt
        """
        dt_f = float(dt)
        v0, st0 = x0[1].vx, x0[1].delta

        def get_digits(val: D) -> int:
            dig = 0
            val %= 1
            while val > 0:
                val = (val * 10) % 1
                dig += 1
            return dig

        # Steady state dynamics - Change velocity and steering at start
        x0[1].vx += u.acc * dt_f
        x0[1].delta += u.ddelta * dt_f
        u0 = VehicleCommands(0.0, 0.0)
        idx = {"x": 0, "y": 1, "th": 2, "v": 3, "st": 4, "ax": 5, "dst": 6} #"t": 5
        digits = get_digits(dt_samp)

        def array_from_state(x_s: VehicleState, u_s: VehicleCommands = u0) -> np.array:
            return np.array([x_s.x, x_s.y, x_s.theta, x_s.vx, x_s.delta, u_s.acc, u_s.ddelta])#float(x_s.t)

        def states_from_array(y: np.array) -> Tuple[VehicleState, VehicleCommands]:
            state = VehicleState(
                x=y[idx["x"]],
                y=y[idx["y"]],
                theta=y[idx["th"]],
                vx=y[idx["v"]],
                delta=y[idx["st"]],
                #t=round(D(y[idx["t"]]), digits),
            )
            action = VehicleCommands(acc=y[idx["ax"]], ddelta=y[idx["dst"]])
            return state, action

        def dynamics(t, y):
            state0, action = states_from_array(y=y)
            rates = self.dynamics(x0=state0, u=action, mean=False)
            return array_from_state(x_s=rates)

        state_i = array_from_state(x_s=x0[1])
        points = int(round(dt / dt_samp, 0)) + 1
        t_eval = np.linspace(0.0, float(dt), points)
        result = solve_ivp(fun=dynamics, t_span=(0.0, float(dt)), y0=state_i, t_eval=t_eval)

        if not result.success:
            raise Exception("Failed to integrate ivp!")

        ret: List[Tuple[Timestamp, VehicleState]] = []
        for time, y_np in zip(result.t, result.y.T):
            state_f, _ = states_from_array(y_np)
            ret.append((time + x0[0], state_f))
        x0[1].vx, x0[1].delta = v0, st0
        return ret[-1], ret

    def dynamics(self, x0: VehicleState, u: VehicleCommands, mean: bool = True) -> VehicleState:
        """Get rate of change of states for given control inputs"""
        dx = x0.vx
        dr = dx * math.tan(x0.delta) / ((self.vg.lf + self.vg.lr)) # todo which lr or lf
        dy = dr * (self.vg.lf + self.vg.lr)/2 # todo which lr og lf
        #dr = dx * math.tan(x0.st) / (2.0 * (self.vg.lf+self.vg.lr))
        #dy = dr * (self.vg.lf+self.vg.lr)
        th_eq = x0.theta + dr / 2.0 if mean else x0.theta
        costh = math.cos(th_eq)
        sinth = math.sin(th_eq)

        xdot = dx * costh - dy * sinth
        ydot = dx * sinth + dy * costh
        ret = VehicleState(x=xdot, y=ydot, theta=dr, vx=u.acc, delta=u.ddelta)#, t=D("1"))
        return ret
