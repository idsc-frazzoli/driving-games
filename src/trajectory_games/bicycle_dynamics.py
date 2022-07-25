import math
from decimal import Decimal as D
from itertools import product
from typing import FrozenSet, List, Set, Tuple

import numpy as np
from commonroad.common.solution import VehicleType, vehicle_parameters
from scipy.integrate import solve_ivp
from vehiclemodels.vehicle_dynamics_ks import vehicle_dynamics_ks

from dg_commons import Timestamp
from dg_commons.sim.models.vehicle import VehicleState, VehicleCommands, VehicleGeometry
from .structures import TrajectoryGenParams

__all__ = ["BicycleDynamics"]


class BicycleDynamics:
    v_max: float
    v_min: float
    """ Maximum and Minimum velocities [m/s] """

    v_switch: float
    """ Switching velocity [m/s]"""

    st_max: float
    """ Maximum steering angle [rad] """

    dst_max: float
    """ Maximum steering rate [rad/s] """

    acc_max: float
    """ Maximum absolute value of acceleration [m/s²]"""

    vg: VehicleGeometry
    """ The vehicle's geometry parameters"""

    u_acc: FrozenSet[float]
    """ Possible values of acceleration [m/s2] """

    u_dst: FrozenSet[float]
    """ Possible values of steering rate [rad/s] """

    def __init__(self, params: TrajectoryGenParams):
        self.v_max = params.v_max
        self.v_min = params.v_min
        self.v_switch = params.v_switch
        self.st_max = params.st_max
        self.dst_max = params.dst_max
        self.acc_max = params.acc_max
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
            res.add(VehicleCommands(acc=acc, ddelta=dst))
        return res

    def get_feasible_acc(self, x: VehicleState, dt: D, u_acc: Set[float]) -> Set[float]:
        """Get feasible accelerations due to vehicle speed limits,
        maximum motor power and max. absolute motor acceleration"""
        dt_f = float(dt)

        # maximal acceleration given by limits on motor speed
        if x.vx > self.v_switch:
            max_acc_motor = self.acc_max * self.v_switch / x.vx
        else:
            max_acc_motor = self.acc_max

        u_acc_clipped = set(
            [
                float(
                    np.clip(
                        a=a,
                        a_min=max((self.v_min - x.vx) / dt_f, -self.acc_max),
                        a_max=min((self.v_max - x.vx) / dt_f, self.acc_max, max_acc_motor),
                    )
                )
                for a in u_acc
            ]
        )
        return u_acc_clipped

    def get_feasible_dst(self, x: VehicleState, dt: D, u_dst: Set[float]) -> Set[float]:
        """Get feasible steering rates due to vehicle steering limits and max steering rate"""
        dt_f = float(dt)
        u_dst_c = set(
            [
                float(np.clip(a=dst, a_min=(-self.st_max - x.delta) / dt_f, a_max=(self.st_max - x.delta) / dt_f))
                for dst in u_dst
            ]
        )

        u_dst_clipped = set([float(np.clip(a=dst, a_min=-self.dst_max, a_max=self.dst_max)) for dst in u_dst_c])
        return u_dst_clipped

    def get_feasible_acc_dst_pairs(
        self, u_acc: Set[float], u_dst: Set[float], x: VehicleState, dt: D
    ) -> Set[Tuple[float, float]]:
        """Get feasible accelerations and steering rates respecting friction circle, motor limits,
        vehicle max and min speed and max and min steering"""

        feasible_pairs = set()

        # clip acceleration due to maximum speed velocities, motor power and max absolute motor acceleration
        u_acc_c = self.get_feasible_acc(x=x, dt=dt, u_acc=u_acc)
        # clip steering rate due to maximum steering angle bound
        u_dst_c = self.get_feasible_dst(x=x, dt=dt, u_dst=u_dst)

        # keep only pairs of acceleration and steering that are inside the friction circle
        dt = float(dt)

        for acc, dst in product(u_acc_c, u_dst_c):
            delta_next = x.delta + dt * dst
            v_next = x.vx + dt * acc
            # pick more conservative case for feasibility
            # if abs(x.vx) > abs(v_next):
            #     v_next = x.vx
            # if abs(x.delta) > abs(delta_next):
            #     delta_next = x.delta

            inside_friction_circle = (
                acc**2 <= self.acc_max**2 - (v_next * v_next * math.tan(delta_next) / self.vg.length) ** 2
            )
            if inside_friction_circle:
                feasible_pairs.add((acc, dst))

        # # find at least a pair that maintains vehicle inside friction circle
        # if feasible_pairs == {}:
        #     inside_friction_circle = acc ** 2 <= self.acc_max ** 2 - (
        #             x.vx * x.vx * math.tan(delta_next) / self.vg.length) ** 2

        return feasible_pairs

    def successor_ivp(
        self, x0: Tuple[Timestamp, VehicleState], u: VehicleCommands, dt: D, dt_samp: D
    ) -> Tuple[Tuple[Timestamp, VehicleState], List[Tuple[Timestamp, VehicleState]]]:
        """
        Perform initial value problem integration
        to propagate state using actions for time dt
        Does not check that accelerations are feasible (in friction circle)
        """
        idx = {"x": 0, "y": 1, "th": 2, "v": 3, "st": 4, "ax": 5, "dst": 6}

        def array_from_state(x_s: VehicleState, u_s: VehicleCommands) -> np.array:
            return np.array([x_s.x, x_s.y, x_s.psi, x_s.vx, x_s.delta, u_s.acc, u_s.ddelta])

        def states_from_array(y: np.array) -> Tuple[VehicleState, VehicleCommands]:
            state = VehicleState(
                x=y[idx["x"]],
                y=y[idx["y"]],
                psi=y[idx["th"]],
                vx=y[idx["v"]],
                delta=y[idx["st"]],
            )
            action = VehicleCommands(acc=y[idx["ax"]], ddelta=y[idx["dst"]])
            return state, action

        def dynamics(t, y):
            state0, action = states_from_array(y=y)
            rates = self.dynamics(x0=state0, u=action)
            return array_from_state(x_s=rates, u_s=VehicleCommands(acc=0.0, ddelta=0.0))

        state_i = array_from_state(x_s=x0[1], u_s=u)
        points = int(round(dt / dt_samp, 0)) + 1
        t_eval = np.linspace(0.0, float(dt), points)
        result = solve_ivp(fun=dynamics, t_span=(0.0, float(dt)), y0=state_i, t_eval=t_eval)

        if not result.success:
            raise Exception("Failed to integrate ivp!")

        ret: List[Tuple[Timestamp, VehicleState]] = []
        for time, y_np in zip(result.t, result.y.T):
            state_f, _ = states_from_array(y_np)
            ret.append((time + x0[0], state_f))

        return ret[-1], ret

    def dynamics(self, x0: VehicleState, u: VehicleCommands) -> VehicleState:
        """Get rate of change of states for given control inputs - new model
        Commonroad model"""

        vehicle_type = VehicleType.FORD_ESCORT
        p = vehicle_parameters[vehicle_type]
        u_init = np.array([u.ddelta, u.acc])

        rate = vehicle_dynamics_ks(np.array([x0.x, x0.y, x0.delta, x0.vx, x0.psi]), u_init, p)

        x_rate = VehicleState(x=rate[0], y=rate[1], psi=rate[4], vx=rate[3], delta=rate[2])
        return x_rate

    # def successors(self, x: VehicleState, dt: D, u0: VehicleCommands = None) -> Mapping[VehicleCommands, VehicleState]:
    #     """For each state, returns a dictionary U -> Possible Xs"""
    #
    #     if u0 is None:
    #         u0 = VehicleCommands(acc=0.0, ddelta=0.0)
    #     u_acc = self.get_feasible_acc(x=x, dt=dt, u0=u0)
    #     u_dst = self.get_feasible_dst(x=x, dt=dt, u0=u0)
    #
    #     res = {}
    #     for acc, dst in product(u_acc, u_dst):
    #         u = VehicleCommands(acc=acc, ddelta=dst)
    #         state, _ = self.successor_forward(x, u, dt)
    #         res[u] = state
    #     return res

    # def successor(self, x0: VehicleState, u: VehicleCommands, dt: D) -> Tuple[VehicleState, List[VehicleState]]:
    #     """Perform RK2 integration to propagate state using actions for time dt"""
    #     dt_f = float(dt)
    #     vf = self.get_clipped(val=x0.vx + u.acc * dt_f, lo=self.v_min, hi=self.v_max)
    #     stf = self.get_clipped(val=x0.delta + u.ddelta * dt_f, lo=-self.st_max, hi=self.st_max)
    #     u_clip = VehicleCommands(acc=(vf - x0.vx) / dt_f, ddelta=(stf - x0.delta) / dt_f)
    #
    #     k1 = self.dynamics(x0, u_clip)
    #     k2 = self.dynamics(x0 + k1 * dt_f, u_clip)
    #     ret = x0 + (k1 + k2) * (dt_f / 2.0)
    #     return ret, [x0, ret]

    # def successor_forward(self, x0: VehicleState, u: VehicleCommands, dt: D) -> Tuple[VehicleState, List[VehicleState]]:
    #     """Perform Euler forward integration to propagate state using actions for time dt"""
    #     dt_f = float(dt)
    #     v0, st0 = x0.v, x0.st
    #     x0.v += u.acc * dt_f
    #     x0.st += u.dst * dt_f
    #     tol = 1e-3
    #     if not self.v_min - tol <= x0.v <= self.v_max + tol:
    #         print("Velocity outside limits")
    #     if not -self.st_max - tol <= x0.st <= self.st_max + tol:
    #         print(f"Steering = {x0.st}, outside limits")
    #     u = VehicleCommands(acc=0.0, dst=0.0)
    #
    #     k1 = self.dynamics(x0, u)
    #     ret = x0 + k1 * dt_f
    #     x0.v, x0.st = v0, st0
    #     return ret, [x0, ret]

    # def max_acc_friction_circle(self, x: VehicleState, u: VehicleCommands) -> Optional[VehicleCommands]:
    #     """
    #     Return clipped vehicle commands that respect friction circle (Kamm's Circle).
    #     a_long² <=  a_max² - (v*theta_dot)²
    #
    #     :param x: current vehicle state
    #     :param u: inputs to apply
    #     :return: inputs with clipped (if necessary) acceleration
    #
    #     """
    #
    #     theta_dot = x.vx * math.tan(x.delta) / self.vg.length
    #     upper_bound_sq = self.acc_max ** 2 - (theta_dot * x.vx) ** 2
    #
    #     if upper_bound_sq < 0.0:
    #         print("Velocity and steering are too high for the bicycle model!")
    #         u.acc = -self.acc_max
    #         return u
    #     else:
    #         ub = math.sqrt(upper_bound_sq)
    #         if abs(u.acc) > ub:
    #             sign = np.sign(u.acc)
    #             u.acc = ub * sign
    #
    #     return u
