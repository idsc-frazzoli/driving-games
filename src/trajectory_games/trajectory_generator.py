import math
from abc import ABC, abstractmethod
from functools import lru_cache
from time import perf_counter
from typing import FrozenSet, Set, List, Dict, Tuple, Mapping
import numpy as np

import geometry as geo
from duckietown_world import relative_pose
from duckietown_world.utils import SE2_apply_R2
from scipy.optimize import minimize

from games import PlayerName
from world import LaneSegmentHashable
from .structures import VehicleState, TrajectoryParams, VehicleActions
from .static_game import StaticActionSetGenerator
from .paths import FinalPoint, Transition, TransitionGraph, Trajectory
from .trajectory_world import TrajectoryWorld
from .bicycle_dynamics import BicycleDynamics

__all__ = ["StaticGenerator", "DynamicGenerator", "TransitionGenerator"]

Successors = Mapping[VehicleActions, Tuple[VehicleState, List[VehicleState]]]


class StaticGenerator(StaticActionSetGenerator[VehicleState, Trajectory, TrajectoryWorld], ABC):
    @abstractmethod
    def get_action_set(self, state: VehicleState, player: PlayerName,
                       world: TrajectoryWorld) -> FrozenSet[Trajectory]:
        pass


class DynamicGenerator(ABC):
    @abstractmethod
    def get_action_tree(self, state: VehicleState, player: PlayerName,
                        world: TrajectoryWorld) -> TransitionGraph:
        pass


class TransitionGenerator(StaticGenerator, DynamicGenerator):
    params: TrajectoryParams
    _bicycle_dyn: BicycleDynamics
    _cache: Dict[Tuple[PlayerName, VehicleState], TransitionGraph]

    def __init__(self, params: TrajectoryParams):
        self.params = params
        self._bicycle_dyn = BicycleDynamics(params=params)
        self._cache = {}

    def get_action_tree(self, state: VehicleState, player: PlayerName,
                        world: TrajectoryWorld = None) -> TransitionGraph:
        """
        Computes many feasible trajectories for given state along reference
        Required world for first instance, returns from cache if already computed
        Updates state graph if provided as input
        """
        if (player, state) in self._cache:
            return self._cache[(player, state)]
        assert world is not None
        tic = perf_counter()
        lane = world.get_lane(player=player)
        graph = TransitionGraph(origin=state)
        self._get_trajectory_graph(state=state, lane=lane, graph=graph)
        toc = perf_counter() - tic
        print(f"Player: {player}\ttime = {toc:.2f} s")
        self._cache[(player, state)] = graph
        return graph

    def get_action_set(self, state: VehicleState, player: PlayerName,
                       world: TrajectoryWorld) -> FrozenSet[Trajectory]:
        tic = perf_counter()
        graph = self.get_action_tree(state=state, player=player, world=world)
        trajectories = self._trajectory_graph_to_list(graph=graph)
        toc = perf_counter() - tic
        if toc > 0.1:
            print(f"Player: {player}\n\tTrajectories generated = {len(trajectories)}\n\ttime = {toc:.2f} s")
        return frozenset(trajectories)

    def _get_trajectory_graph(self, state: VehicleState, lane: LaneSegmentHashable, graph: TransitionGraph):
        """ Construct graph of states """
        stack = list([state])
        graph.origin = state

        p_final = self.get_p_final(lane=lane)
        if p_final is not None:
            x_f, y_f, inc = p_final
            z_f = x_f if x_f is not None else y_f
        else:
            x_f, z_f, inc = None, None, True

        if state not in graph.nodes:
            graph.add_node(state=state, gen=0)
        expanded = set()
        while stack:
            s1 = stack.pop(0)
            assert s1 in graph.nodes
            if s1 in expanded:
                continue
            n_gen = graph.nodes[s1]["gen"]
            expanded.add(s1)
            successors = self.tree_func(state=s1, lane=lane, gen=n_gen)
            for u, (s2, samp) in successors.items():
                if p_final is not None:
                    z = s2.x if x_f is not None else s2.y
                    cond = (inc and (z < z_f)) or (not inc and (z > z_f))
                else:
                    cond = n_gen + 1 < self.params.max_gen
                if cond:
                    stack.append(s2)
                transition = Transition.create(states=(s1, s2), p_final=p_final,
                                               sampled=samp)
                graph.add_edge(transition=transition, u=u)

        return graph

    @staticmethod
    def get_curv(state: VehicleState, lane: LaneSegmentHashable) -> Tuple[float, float, float]:
        """ Calculate curvilinear coordinates for state """
        p = np.array([state.x, state.y])
        q = geo.SE2_from_translation_angle(t=p, theta=state.th)

        beta, q0 = lane.find_along_lane_closest_point(p=p)
        along = lane.along_lane_from_beta(beta)
        rel = relative_pose(q0, q)
        r, mu, _ = geo.translation_angle_scale_from_E2(rel)
        return along, r[1], mu

    @staticmethod
    def get_target(lane: LaneSegmentHashable, progress: float,
                   offset_target: np.array) -> Tuple[np.array, float]:
        """ Calculate target pose ([x, y], theta) at requested progress with additional offset """
        beta_f = lane.beta_from_along_lane(along_lane=progress)
        q_f = lane.center_point(beta=beta_f)
        _, ang_f, _ = geo.translation_angle_scale_from_E2(q_f)
        pos_f = SE2_apply_R2(q_f, offset_target)
        return pos_f, ang_f

    def get_successor(self, state: VehicleState, u: VehicleActions, samp: bool = True) \
            -> Tuple[VehicleState, List[VehicleState]]:
        dt_samp = self.params.dt_samp if samp else self.params.dt
        return self._bicycle_dyn.successor_ivp(x0=state, u=u, dt=self.params.dt,
                                               dt_samp=dt_samp)

    def tree_func(self, state: VehicleState, lane: LaneSegmentHashable,
                  gen: int) -> Successors:
        if self.params.solve:
            return self.get_successors_solve(state=state, lane=lane, gen=gen)
        else:
            return self.get_successors_approx(state=state, lane=lane, gen=gen)

    def get_acc_dst(self, state: VehicleState, gen: int) -> Tuple[Set[float], Set[float]]:
        u0 = VehicleActions(acc=0.0, dst=0.0)
        cond_gen = gen < self.params.max_gen
        dst_vals = self._bicycle_dyn.u_dst if cond_gen else {0.0}
        acc_vals = self._bicycle_dyn.get_feasible_acc(x=state, dt=self.params.dt, u0=u0)
        if not cond_gen:
            for acc in list(acc_vals):
                if acc < 0.0:
                    acc_vals.remove(acc)
                    acc_vals.add(0.0)
        return acc_vals, dst_vals

    def get_successors_approx(self, state: VehicleState, lane: LaneSegmentHashable,
                              gen: int) -> Successors:
        """
        Approximate method to grow trajectory tree (fast)
        Predicts progress along reference using curvature
        Samples discrete grid of progress (from acceleration) and deviation
        Steers car using kinematic model to reach close to target point
        """

        dt = float(self.params.dt)
        l = self.params.vg.l

        # Calculate initial pose
        start_arr = np.array([state.x, state.y])
        th_start = state.th
        along_i, n_i, mui = self.get_curv(state=state, lane=lane)

        # Calculate real axle translation and rotation
        offset_0, offset_i = np.array([0, 0]), np.array([-l, 0])
        p_i, th_i = self.get_target(lane=lane, progress=along_i, offset_target=offset_0)
        q_start = geo.SE2_from_translation_angle(t=start_arr, theta=state.th)
        p_start = SE2_apply_R2(q_start, offset_i)

        def get_progress(acc: float, K: float) -> float:
            """ Progress along reference using curvature"""
            vf = state.v + acc * dt
            return (vf * dt) / (1 - n_i * K)

        def get_corrected_distance(acc: float) -> float:
            """ Progress along reference iteratively corrected using curvature"""
            curv = 0.0
            dist = get_progress(acc=acc, K=curv)
            for i in range(5):
                p_f, th_f = self.get_target(lane=lane, progress=along_i + dist,
                                            offset_target=offset_0)
                dlb = p_f - p_i
                Lb = np.linalg.norm(dlb)

                # Two points with heading average curvature computation
                curv = 2 * math.sin(th_f - th_i) / Lb
                dist_new = get_progress(acc=acc, K=curv)
                if abs(dist - dist_new) < 0.1:
                    dist = dist_new
                    break
                dist = dist_new

            return dist

        st_max, dst_max = self.params.st_max, self.params.dst_max
        acc_vals, dst_vals = self.get_acc_dst(state=state, gen=gen)
        successors: Dict[VehicleActions, Tuple[VehicleState, List[VehicleState]]] = {}

        # Sample progress using acceleration
        for accel in acc_vals:
            distance = get_corrected_distance(acc=accel)

            # Sample deviation as a function of dst
            for dst in dst_vals:
                # Calculate target pose of rear axle
                nf = 0.5 * (n_i + dst * distance)
                offset_t = np.array([-l, nf])
                p_t, th_t = self.get_target(lane=lane, progress=along_i + distance,
                                            offset_target=offset_t)

                # Steer from initial to final position using kinematic model
                #  No slip at rear axle assumption --> Rear axle moves along a circle
                dlb_t = p_t - p_start
                Lb_t = np.linalg.norm(dlb_t)
                alpb = math.atan2(dlb_t[1], dlb_t[0]) - th_start
                tan_st = 4 * math.sin(alpb) * l / Lb_t
                st_f = min(max(math.atan(tan_st), -st_max), st_max)
                dst_f = min(max((st_f - state.st) / dt, -dst_max), dst_max)

                # Propagate inputs to obtain exact final state
                u = VehicleActions(acc=accel, dst=dst_f)
                state_f, states_t = self.get_successor(state=state, u=u)
                successors[u] = (state_f, states_t)

        return successors

    def get_successors_solve(self, state: VehicleState, lane: LaneSegmentHashable,
                             gen: int) -> Successors:
        """
        Accurate method to grow trajectory tree (slow)
        Samples discrete grid of velocity (from acceleration) and deviation
        Solves a two point boundary value problem to calculate steering angle
        Propagates states using calculated steering and kinematic model
        """

        dt = float(self.params.dt)
        s_init, n_init, mui = self.get_curv(state=state, lane=lane)
        successors: Dict[VehicleActions, Tuple[VehicleState, List[VehicleState]]] = {}

        # Steering rate bounds
        dst_max = self.params.dst_max
        lb = max(-dst_max, (-self.params.st_max - state.st) / dt)
        ub = min(+dst_max, (+self.params.st_max - state.st) / dt)
        acc_vals, dst_vals = self.get_acc_dst(state=state, gen=gen)

        def equation_forward(vars_in, acc: float) -> Tuple[float, float]:
            """ Euler forward integration (cartesian) to obtain curvilinear state """
            u = VehicleActions(acc=acc, dst=vars_in[0])
            state_end, _ = self.get_successor(state=state, u=u, samp=False)
            _, n, mu = self.get_curv(state=state_end, lane=lane)
            return n, mu

        def equation_min(vars_in, acc: float, nfinal: float) -> float:
            """ Function for optimiser """
            n, mu = equation_forward(vars_in, acc=acc)
            return (n - nfinal) ** 2 + float(np.abs(mu) > np.pi / 2) * 10000

        def get_dst_guess() -> float:
            """ Initial guess for optimisation, obtained from target yaw rate """
            p_t, th_t = self.get_target(lane=lane, progress=s_init + distance,
                                        offset_target=np.array([0, 0]))
            d_ang = (th_t - state.th)
            while d_ang > +np.pi: d_ang -= 2 * np.pi
            while d_ang < -np.pi: d_ang += 2 * np.pi
            dst_i = (math.atan(d_ang * 2 * self.params.vg.l / state.v * dt) - state.st) / dt
            dst_i = min(max(dst_i, lb), ub)
            return dst_i

        # Sample velocities
        for accel in acc_vals:
            vf = state.v + accel * dt
            distance = vf * dt

            # Sample deviations
            for dst in dst_vals:
                nf = 0.5 * (n_init + dst * distance)

                # Solve boundary value problem to obtain actions
                dst_g = get_dst_guess()
                res = minimize(fun=equation_min, x0=np.array([dst_g]),
                               bounds=[[lb, ub]], args=(accel, nf))
                i = 0
                dst0 = [0.0, lb / 2, ub / 2]

                # Try other initial states if it doesn't converge
                while not res.success and i <= 2:
                    res = minimize(fun=equation_min, x0=np.array([dst0[i]]),
                                   bounds=[[lb, ub]], args=(accel, nf))
                    i += 1
                if not res.success:
                    print(f"Opt failed: {state}, acc={accel}, nf={nf}")
                    continue
                dst = res.x[0]

                # Propagate inputs to obtain final state
                u_f = VehicleActions(acc=accel, dst=dst)
                state_f, states_t = self.get_successor(state=state, u=u_f)
                successors[u_f] = (state_f, states_t)
        return successors

    @lru_cache(None)
    def get_p_final(self, lane: LaneSegmentHashable) -> FinalPoint:
        if self.params.s_final < 0:
            return None
        tol = 1e-1
        s_max = lane.get_lane_length()
        s_final = s_max * self.params.s_final
        beta_final = lane.beta_from_along_lane(along_lane=s_final)
        center = lane.center_point(beta=beta_final)
        pos_f, ang_f, _ = geo.translation_angle_scale_from_E2(center)
        while ang_f < -math.pi: ang_f += 2 * math.pi
        while ang_f > +math.pi: ang_f -= 2 * math.pi
        if abs(ang_f) < tol:
            p_f = (pos_f[0], None, True)
        elif abs(ang_f - math.pi) < tol or abs(ang_f + math.pi) < tol:
            p_f = (pos_f[0], None, False)
        elif abs(ang_f - math.pi / 2) < tol:
            p_f = (None, pos_f[1], True)
        elif abs(ang_f + math.pi / 2) < tol:
            p_f = (None, pos_f[1], False)
        else:
            raise Exception("Final angle is not along axes!")
        return p_f

    @staticmethod
    def _trajectory_graph_to_list(graph: TransitionGraph) -> Set[Trajectory]:
        """ Convert state graph to list of trajectories"""
        trajectories: Set[Trajectory] = set()
        roots = [n for n, d in graph.in_degree() if d == 0]
        assert len(roots) == 1
        source = roots[0]
        leaves = [n for n, d in graph.out_degree() if d == 0]

        for target in leaves:
            trajectories.add(graph.get_trajectory(source=source, target=target))
        return trajectories
