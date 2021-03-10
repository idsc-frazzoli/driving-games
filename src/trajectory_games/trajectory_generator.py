import math
from abc import ABC, abstractmethod
from time import perf_counter
from typing import FrozenSet, Set, List, Dict, Tuple, Mapping
import numpy as np
from decimal import Decimal as D

import geometry as geo
from duckietown_world import relative_pose
from duckietown_world.utils import SE2_apply_R2
from networkx import MultiDiGraph, topological_sort
from scipy.optimize import minimize

from games import PlayerName
from world import LaneSegmentHashable
from .structures import VehicleState, TrajectoryParams, VehicleActions
from .static_game import ActionSetGenerator
from .paths import Trajectory
from .trajectory_world import TrajectoryWorld
from .bicycle_dynamics import BicycleDynamics

__all__ = ["TrajectoryGenerator", "TrajectoryGenerator1"]


class TrajectoryGenerator(ActionSetGenerator[VehicleState, Trajectory, TrajectoryWorld], ABC):
    @abstractmethod
    def get_action_set(self, state: VehicleState, player: PlayerName, world: TrajectoryWorld, **kwargs) -> FrozenSet[Trajectory]:
        pass


class TrajectoryGenerator1(TrajectoryGenerator):
    params: TrajectoryParams
    _bicycle_dyn: BicycleDynamics
    _cache: Dict[Tuple[PlayerName, VehicleState], FrozenSet[Trajectory]]

    def __init__(self, params: TrajectoryParams):
        self.params = params
        self._bicycle_dyn = BicycleDynamics(params=params)
        self._cache = {}

    def get_action_set(self, state: VehicleState, player: PlayerName,
                       world: TrajectoryWorld = None, graph: MultiDiGraph = None) \
            -> FrozenSet[Trajectory]:
        """
        Computes many feasible trajectories for given state along reference
        Required world for first instance, returns from cache if already computed
        Updates state graph if provided as input
        """
        if (player, state) in self._cache:
            return self._cache[(player, state)]
        assert world is not None
        tic = perf_counter()
        G = self._get_trajectory_graph(state=state, lane=world.get_lane(player=player))
        if isinstance(graph, MultiDiGraph):
            graph.__init__(G)
        trajectories = self._trajectory_graph_to_list(G=G, dt=self.params.dt_samp)
        toc = perf_counter() - tic
        print(f"Trajectory generation time = {toc:.2f} s")
        ret = frozenset(trajectories)
        self._cache[(player, state)] = ret
        return ret

    def _get_trajectory_graph(self, state: VehicleState, lane: LaneSegmentHashable) -> MultiDiGraph:
        """ Construct graph of states """
        stack = list([state])
        G = MultiDiGraph()

        def add_node(s, gen):
            G.add_node(s, gen=gen, x=s.x, y=s.y)

        add_node(state, gen=0)
        i: int = 0
        expanded = set()
        while stack:
            i += 1
            s1 = stack.pop(0)
            assert s1 in G.nodes
            if s1 in expanded:
                continue
            n_gen = G.nodes[s1]["gen"]
            expanded.add(s1)
            successors = self.get_successors_solve(state=s1, lane=lane) if self.params.solve \
                else self.get_successors_approx(state=s1, lane=lane)
            for u, s2 in successors.items():
                if s2 not in G.nodes:
                    add_node(s2, gen=n_gen + 1)
                    if n_gen + 1 < self.params.max_gen:
                        stack.append(s2)
                G.add_edge(s1, s2, u=u, gen=n_gen)

        return G

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

    def get_successors_approx(self, state: VehicleState, lane: LaneSegmentHashable) -> \
            Mapping[VehicleActions, VehicleState]:
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
                p_f, th_f = self.get_target(lane=lane, progress=along_i+dist, offset_target=offset_0)
                dlb = p_f - p_i
                Lb = np.linalg.norm(dlb)

                # Two points with heading average curvature computation
                curv = 2 * math.sin(th_f - th_i) / Lb
                dist_new = get_progress(acc=acc, K=curv)
                if abs(dist-dist_new) < 0.1:
                    dist = dist_new
                    break
                dist = dist_new

            return dist

        st_max, dst_max = self.params.st_max, self.params.dst_max
        successors: Dict[VehicleActions, VehicleState] = {}
        u0 = VehicleActions(acc=0.0, dst=0.0)

        # Sample progress using acceleration
        for accel in self._bicycle_dyn.get_feasible_acc(x=state, dt=self.params.dt, u0=u0):
            distance = get_corrected_distance(acc=accel)

            # Sample deviation as a function of dst
            for dst in self._bicycle_dyn.u_dst:

                # Calculate target pose of rear axle
                nf = n_i * 0.5 + dst * distance / l
                offset_t = np.array([-l, nf])
                p_t, th_t = self.get_target(lane=lane, progress=along_i+distance, offset_target=offset_t)

                # Steer from initial to final position using kinematic model
                #  No slip at rear axle assumption --> Rear axle moves along a circle
                dlb_t = p_t - p_start
                Lb_t = np.linalg.norm(dlb_t)
                alpb = math.atan2(dlb_t[1], dlb_t[0]) - th_start
                tan_st = 4 * math.sin(alpb) * dt * l / Lb_t
                st_f = min(max(math.atan(tan_st), -st_max), st_max)
                dst_f = min(max((st_f - state.st) / dt, -dst_max), dst_max)

                # Propagate inputs to obtain exact final state
                u = VehicleActions(acc=accel, dst=dst_f)
                state_f = self._bicycle_dyn.successor_forward(x0=state, u=u, dt=self.params.dt)
                successors[u] = state_f

        return successors

    def get_successors_solve(self, state: VehicleState, lane: LaneSegmentHashable) -> \
            Mapping[VehicleActions, VehicleState]:
        """
        Accurate method to grow trajectory tree (slow)
        Samples discrete grid of velocity (from acceleration) and deviation
        Solves a two point boundary value problem to calculate steering angle
        Propagates states using calculated steering and kinematic model
        """

        dt = float(self.params.dt)
        s_init, n_init, mui = self.get_curv(state=state, lane=lane)
        successors: Dict[VehicleActions, VehicleState] = {}

        # Steering rate bounds
        dst_max = self.params.dst_max
        lb = max(-dst_max, (-self.params.st_max - state.st) / dt)
        ub = min(+dst_max, (+self.params.st_max - state.st) / dt)
        u0 = VehicleActions(acc=0.0, dst=0.0)

        def equation_forward(vars_in, acc: float) -> Tuple[float, float]:
            """ Euler forward integration (cartesian) to obtain curvilinear state """
            u = VehicleActions(acc=acc, dst=vars_in[0])
            state_end = self._bicycle_dyn.successor_forward(x0=state, u=u, dt=self.params.dt)
            _, n, mu = self.get_curv(state=state_end, lane=lane)
            return n, mu

        def equation_min(vars_in, acc: float, nfinal: float) -> float:
            """ Function for optimiser """
            n, mu = equation_forward(vars_in, acc=acc)
            return (n - nfinal) ** 2 + float(np.abs(mu) > np.pi / 2) * 10000

        def get_dst_guess() -> float:
            """ Initial guess for optimisation, obtained from target yaw rate """
            p_t, th_t = self.get_target(lane=lane, progress=s_init+distance, offset_target=np.array([0, 0]))
            d_ang = (th_t - state.th)
            while d_ang > +np.pi: d_ang -= 2*np.pi
            while d_ang < -np.pi: d_ang += 2*np.pi
            dst_i = (math.atan(d_ang * 2 * self.params.vg.l / state.v*dt) - state.st) / dt
            dst_i = min(max(dst_i, lb), ub)
            return dst_i

        # Sample velocities
        for accel in self._bicycle_dyn.get_feasible_acc(x=state, dt=self.params.dt, u0=u0):
            vf = state.v + accel * dt
            distance = vf * dt

            # Sample deviations
            for dst in self._bicycle_dyn.u_dst:
                nf = n_init * 0.5 + (dst/self.params.vg.l) * distance

                # Solve boundary value problem to obtain actions
                dst_g = get_dst_guess()
                res = minimize(fun=equation_min, x0=np.array([dst_g]),
                               bounds=[[lb, ub]], args=(accel, nf))
                i = 0
                dst0 = [0.0, lb/2, ub/2]

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
                state_f = self._bicycle_dyn.successor_forward(x0=state, u=u_f, dt=self.params.dt)
                successors[u_f] = state_f
        return successors

    @staticmethod
    def _trajectory_graph_to_list(G: MultiDiGraph, dt: D) -> Set[Trajectory]:
        """ Convert state graph to list of trajectories"""
        expanded = set()
        all_traj: Set[Trajectory] = set()
        for n1 in topological_sort(G):
            traj: List[VehicleState] = [n1]
            TrajectoryGenerator1._expand_graph(G=G, node=n1, traj=traj,
                                               all_traj=all_traj,
                                               expanded=expanded, dt=dt)
        return all_traj

    @staticmethod
    def _expand_graph(G: MultiDiGraph, node: VehicleState,
                      traj: List[VehicleState], all_traj: Set[Trajectory],
                      expanded: Set[VehicleState], dt: D):
        """ Recursively expand graph to obtain states """
        if node in expanded:
            return
        successors = list(G.successors(node))
        if not successors:
            all_traj.add(Trajectory(traj=traj, dt_samp=dt))
        else:
            for n2 in successors:
                traj1: List[VehicleState] = traj + [n2]
                TrajectoryGenerator1._expand_graph(G=G, node=n2, traj=traj1,
                                                   all_traj=all_traj,
                                                   expanded=expanded, dt=dt)
        expanded.add(node)
