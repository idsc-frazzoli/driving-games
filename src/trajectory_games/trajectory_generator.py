import math
from time import perf_counter
from typing import Dict, FrozenSet, List, Mapping, Optional, Set, Tuple

import numpy as np
from scipy.optimize import minimize

import geometry as geo
from dg_commons import PlayerName, relative_pose, SE2_apply_T2
from dg_commons.maps import DgLanelet
from .bicycle_dynamics import BicycleDynamics
from .game_def import ActionSetGenerator
from .paths import Trajectory, TrajectoryGraph
from .structures import TrajectoryParams, VehicleActions, VehicleState
from .trajectory_world import TrajectoryWorld

__all__ = ["TransitionGenerator"]

Successors = Mapping[VehicleActions, Tuple[VehicleState, List[VehicleState]]]
Solve_Tolerance = 1e-3


class TransitionGenerator(ActionSetGenerator[VehicleState, Trajectory, TrajectoryWorld]):
    """Generate feasible trajectories for each player"""

    params: TrajectoryParams

    """ Internal data """
    _bicycle_dyn: BicycleDynamics
    _cache: Dict[Tuple[PlayerName, VehicleState], Set[TrajectoryGraph]]

    def __init__(self, params: TrajectoryParams):
        self.params = params
        self._bicycle_dyn = BicycleDynamics(params=params)
        self._cache = {}

    def get_actions_dynamic(
        self, state: VehicleState, player: PlayerName, world: TrajectoryWorld = None
    ) -> Tuple[bool, Set[TrajectoryGraph]]:
        """
        Computes dynamic graph of transitions for given state along reference
        Requires world for first instance, returns from cache if already computed
        """
        if (player, state) in self._cache:
            return True, self._cache[(player, state)]
        assert world is not None
        tic = perf_counter()
        all_graphs: Set[TrajectoryGraph] = set()
        for lane, goal in world.get_lanes(player=player):
            graph = TrajectoryGraph(origin=state, lane=lane, goal=goal)
            self._get_trajectory_graph(state=state, lane=lane, graph=graph)
            all_graphs.add(graph)
        toc = perf_counter() - tic
        # print(f"Player: {player}\ttime = {toc:.2f} s")
        self._cache[(player, state)] = all_graphs
        return False, all_graphs

    def get_actions_static(
        self, state: VehicleState, player: PlayerName, world: TrajectoryWorld = None
    ) -> FrozenSet[Trajectory]:
        """
        Computes set of static feasible trajectories for given state along reference lanes
        Requires world for first instance, returns from cache if already computed
        """
        tic = perf_counter()
        cache, lane_graphs = self.get_actions_dynamic(state=state, player=player, world=world)
        all_traj: Set[Trajectory] = set()
        for graph in lane_graphs:
            all_traj |= self._trajectory_graph_to_list(graph=graph)
        toc = perf_counter() - tic
        if not cache:
            print(
                f"Player: {player}\n\tLanes = {len(lane_graphs)}"
                f"\n\tTrajectories generated = {len(all_traj)}\n\ttime = {toc:.2f} s"
            )
        return frozenset(all_traj)

    def _get_trajectory_graph(self, state: VehicleState, lane: DgLanelet, graph: TrajectoryGraph):
        """Construct graph of states"""
        k_maxgen = 5
        stack = list([state])
        graph.origin = state

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
                if graph.goal is not None:
                    cond = Trajectory.get_in_goal_index(states=samp, goal=graph.goal) is None and n_gen + 1 < k_maxgen
                else:
                    cond = n_gen + 1 < self.params.max_gen
                if cond:
                    stack.append(s2)
                transition = Trajectory.create(values=samp, lane=lane, goal=graph.goal, states=(s1, s2))
                graph.add_edge(trajectory=transition, u=u)

        return graph

    @staticmethod
    def get_curv(state: VehicleState, lane: DgLanelet) -> Tuple[float, float, float]:
        """Calculate curvilinear coordinates for state"""
        p = np.array([state.x, state.y])
        q = geo.SE2_from_translation_angle(t=p, theta=state.th)

        beta, q0 = lane.find_along_lane_closest_point(p=p)
        along = lane.along_lane_from_beta(beta)
        rel = relative_pose(q0, q)
        r, mu, _ = geo.translation_angle_scale_from_E2(rel)
        return along, r[1], mu

    @staticmethod
    def get_target(lane: DgLanelet, progress: float, offset_target: np.array) -> Optional[Tuple[np.array, float]]:
        """Calculate target pose ([x, y], theta) at requested progress with additional offset"""
        beta_f = lane.beta_from_along_lane(along_lane=progress)
        q_f = lane.center_point(beta=beta_f)
        _, ang_f, _ = geo.translation_angle_scale_from_E2(q_f)
        pos_f = SE2_apply_T2(q_f, offset_target)
        return pos_f, ang_f

    def get_successor(
        self, state: VehicleState, u: VehicleActions, samp: bool = True
    ) -> Tuple[VehicleState, List[VehicleState]]:
        dt_samp = self.params.dt_samp if samp else self.params.dt
        return self._bicycle_dyn.successor_ivp(x0=state, u=u, dt=self.params.dt, dt_samp=dt_samp)

    def tree_func(self, state: VehicleState, lane: DgLanelet, gen: int) -> Successors:
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

    def get_successors_approx(self, state: VehicleState, lane: DgLanelet, gen: int) -> Successors:
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
        p_start = SE2_apply_T2(q_start, offset_i)

        def get_progress(acc: float, K: float) -> float:
            """Progress along reference using curvature"""
            vf = state.v + acc * dt
            return (vf * dt) / (1 - n_i * K)

        def get_corrected_distance(acc: float) -> float:
            """Progress along reference iteratively corrected using curvature"""
            curv = 0.0
            dist = get_progress(acc=acc, K=curv)
            for i in range(5):
                p_f, th_f = self.get_target(lane=lane, progress=along_i + dist, offset_target=offset_0)
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
            n_scale = distance if self.params.dst_scale else 1.0

            # Sample deviation as a function of dst
            for dst in dst_vals:
                # Calculate target pose of rear axle
                nf = self.params.n_factor * n_i + dst * n_scale
                offset_t = np.array([-l, nf])
                p_t, th_t = self.get_target(lane=lane, progress=along_i + distance, offset_target=offset_t)

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

    def get_successors_solve(self, state: VehicleState, lane: DgLanelet, gen: int) -> Successors:
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
            """Euler forward integration (cartesian) to obtain curvilinear state"""
            u = VehicleActions(acc=acc, dst=vars_in[0])
            state_end, _ = self.get_successor(state=state, u=u, samp=False)
            _, n, mu = self.get_curv(state=state_end, lane=lane)
            return n, mu

        def equation_min(vars_in, acc: float, nfinal: float) -> float:
            """Function for optimiser"""
            n, mu = equation_forward(vars_in, acc=acc)
            return (n - nfinal) ** 2 + float(np.abs(mu) > np.pi / 2) * 10000

        def get_dst_guess() -> float:
            """Initial guess for optimisation, obtained from target yaw rate"""
            p_t, th_t = self.get_target(lane=lane, progress=s_init + distance, offset_target=np.array([0, 0]))
            d_ang = th_t - state.th
            while d_ang > +np.pi:
                d_ang -= 2 * np.pi
            while d_ang < -np.pi:
                d_ang += 2 * np.pi
            dst_i = (math.atan(d_ang * 2 * self.params.vg.l / state.v * dt) - state.st) / dt
            dst_i = min(max(dst_i, lb), ub)
            return dst_i

        # Sample velocities
        for accel in acc_vals:
            vf = state.v + accel * dt
            distance = vf * dt
            n_scale = distance if self.params.dst_scale else 1.0

            # Sample deviations
            for dst in dst_vals:
                nf = self.params.n_factor * n_init + dst * n_scale

                # Solve boundary value problem to obtain actions
                residual, dst_f = 100.0, 0.0
                # Solution is sensitive to init guess, so try a few values and give up if it doesn't converge
                for dst_g in [0.0, lb / 2, ub / 2, get_dst_guess()]:
                    result = minimize(fun=equation_min, x0=np.array([dst_g]), bounds=[[lb, ub]], args=(accel, nf))
                    if result.success and result.fun < residual:
                        residual = result.fun
                        dst_f = result.x[0]
                    if residual < Solve_Tolerance:
                        break

                if residual >= Solve_Tolerance:
                    # print(f"Opt failed: {state}, acc={accel}, nf={nf}")
                    continue

                # Propagate inputs to obtain final state
                u_f = VehicleActions(acc=accel, dst=dst_f)
                state_f, states_t = self.get_successor(state=state, u=u_f)
                successors[u_f] = (state_f, states_t)
        return successors

    @staticmethod
    def _trajectory_graph_to_list(graph: TrajectoryGraph) -> Set[Trajectory]:
        """Convert state graph to list of trajectories"""
        trajectories: Set[Trajectory] = set()
        roots = [n for n, d in graph.in_degree() if d == 0]
        assert len(roots) == 1
        source = roots[0]
        leaves = [n for n, d in graph.out_degree() if d == 0]

        for target in leaves:
            trajectories.add(graph.get_trajectory(source=source, target=target))
        return trajectories
