import math
from time import perf_counter
from typing import Dict, FrozenSet, List, Mapping, Optional, Set, Tuple, Sequence, Union

import geometry as geo
import numpy as np
from scipy.optimize import minimize

from dg_commons import relative_pose, SE2_apply_T2, Timestamp, SE2Transform
from dg_commons.maps import DgLanelet
from dg_commons.planning import Trajectory, TrajectoryGraph, TimedVehicleState, RefLaneGoal
from dg_commons.sim.models.vehicle import VehicleState, VehicleCommands
from .bicycle_dynamics import BicycleDynamics
from .game_def import ActionSetGenerator


from .structures import TrajectoryGenParams

__all__ = ["TrajectoryGenerator"]

Successors = Mapping[VehicleCommands, Tuple[TimedVehicleState, List[TimedVehicleState]]]
Solve_Tolerance = 10.0 #1e-3


class TrajectoryGenerator(ActionSetGenerator[VehicleState, Trajectory]):
    """Generate feasible trajectories for each player"""

    params: TrajectoryGenParams

    """ Internal data """
    _bicycle_dyn: BicycleDynamics

    # _cache: Dict[Tuple[PlayerName, VehicleState], Set[TrajectoryGraph]]

    def __init__(self, params: TrajectoryGenParams, ref_lane_goals: Sequence[RefLaneGoal]):
        self.params = params
        self.ref_lane_goals: Sequence[RefLaneGoal] = ref_lane_goals
        self._bicycle_dyn = BicycleDynamics(params=params)


    def get_lanes_actions(self, state: VehicleState) -> Set[TrajectoryGraph]:
        """
        Computes dynamic graph of transitions starting from state and for each reference lane
        """
        # tic = perf_counter()
        all_graphs: Set[TrajectoryGraph] = set()
        for ref_lane_goal in self.ref_lane_goals:
            graph = self._get_trajectory_graph(state=state, ref_lane_goal=ref_lane_goal)
            all_graphs.add(graph)
        # toc = perf_counter() - tic
        # print(f"Generating trajectory graphs for player: {player}\ttook: {toc:.2f} s")
        # self._cache[(player, state)] = all_graphs
        return all_graphs

    def get_actions(self, state: VehicleState, return_graphs=False) -> FrozenSet[Union[Trajectory, TrajectoryGraph]]:
        """
        Computes set of feasible trajectories for given state along reference lanes
        """
        tic = perf_counter()
        lane_graphs = self.get_lanes_actions(state=state)
        if return_graphs:
            graphs: List[TrajectoryGraph] = []
            for graph in lane_graphs:
                graphs.append(graph)
            ret = graphs
            toc = perf_counter() - tic
            print(
                f"Lanes = {len(lane_graphs)}" f"\n\tGraphs generated = {len(ret)}\n\ttime = {toc:.2f} s"
            )
        else:
            all_trajs: Set[Trajectory] = set()
            for graph in lane_graphs:
                all_trajs |= graph.get_all_transitions()
            ret = all_trajs
            toc = perf_counter() - tic
            print(
                f"Lanes = {len(lane_graphs)}" f"\n\tTrajectories generated = {len(ret)}\n\ttime = {toc:.2f} s"
            )
        return frozenset(ret)

    def _get_trajectory_graph(self, state: VehicleState, ref_lane_goal: RefLaneGoal) -> TrajectoryGraph:
        """Construct graph of states"""
        graph: TrajectoryGraph = TrajectoryGraph()
        k_maxgen = 5
        t_init: Timestamp = 0.0
        init_state = (t_init, state)
        stack: List[TimedVehicleState] = list([init_state])  # use timed state. Initial time is 0.

        # add root of graph
        if init_state not in graph.nodes:
            graph.add_node(timed_state=init_state, gen=0)
        expanded = set()
        while stack:
            s1 = stack.pop(0)
            assert s1 in graph.nodes
            if s1 in expanded:
                continue
            current_gen = graph.nodes[s1]["gen"]
            expanded.add(s1)
            successors = self.tree_func(timed_state=s1, lane=ref_lane_goal.ref_lane, gen=current_gen)
            for u, (s2, samp) in successors.items():
                if ref_lane_goal.goal_progress is not None:
                    # use finite distance condition
                    # cond = self.get_goal_reached_index(states=samp, goal=goal) is None and current_gen + 1 < k_maxgen
                    cond = current_gen + 1 < k_maxgen and not self.goal_reached(
                        states=samp, ref_lane_goal=ref_lane_goal
                    )
                else:
                    # use finite time condition (time = generations)
                    cond = current_gen + 1 < self.params.max_gen

                if cond:
                    stack.append(s2)
                trans_values = [val[1] for val in samp]
                trans_timestamps = [val[0] for val in samp]
                transition = Trajectory(values=trans_values, timestamps=trans_timestamps)
                # this to add commands that have same size as transition
                # cmds = [u for _ in range(len(trans_timestamps))]
                # commands = Trajectory(values=cmds, timestamps=trans_timestamps)

                timestamps = [s1[0], s2[0]]
                values = [s1[1], s2[1]]
                states = Trajectory(values=values, timestamps=timestamps)

                # this to add a single command (it is constant anyways)
                # commands = (Trajectory(values=[u], timestamps=[s1[0]]))
                if cond:
                    graph.add_edge(states=states, transition=transition, commands=u)

        return graph

    @staticmethod
    def goal_reached(states: List[TimedVehicleState], ref_lane_goal: RefLaneGoal):
        ref_lane = ref_lane_goal.ref_lane
        se2_transforms = [SE2Transform(p=np.array([x[1].x, x[1].y]), theta=x[1].theta) for x in states]
        ref_lane_progress = [ref_lane.lane_pose_from_SE2Transform(q).along_lane for q in se2_transforms]

        return any(progress >= ref_lane_goal.goal_progress for progress in ref_lane_progress)

    @staticmethod
    def get_curv(state: VehicleState, lane: DgLanelet) -> Tuple[float, float, float]:
        """Calculate curvilinear coordinates for state"""
        p = np.array([state.x, state.y])
        q = geo.SE2_from_translation_angle(t=p, theta=state.theta)

        beta, q0 = lane.find_along_lane_closest_point(p=p)
        along = lane.along_lane_from_beta(beta)
        rel = relative_pose(q0, q)
        r, mu, _ = geo.translation_angle_scale_from_E2(rel)
        return along, r[1], mu

    @staticmethod
    def _get_target(lane: DgLanelet, progress: float, offset_target: np.array) -> Optional[Tuple[np.array, float]]:
        """Calculate target pose ([x, y], theta) at requested progress with additional offset"""
        beta_f = lane.beta_from_along_lane(along_lane=progress)
        q_f = lane.center_point(beta=beta_f)
        _, ang_f, _ = geo.translation_angle_scale_from_E2(q_f)
        pos_f = SE2_apply_T2(q_f, offset_target)
        return pos_f, ang_f

    def get_successor(
        self, state: TimedVehicleState, u: VehicleCommands, samp: bool = True
    ) -> Tuple[TimedVehicleState, List[TimedVehicleState]]:
        dt_samp = self.params.dt_samp if samp else self.params.dt
        return self._bicycle_dyn.successor_ivp(x0=state, u=u, dt=self.params.dt, dt_samp=dt_samp)

    def tree_func(self, timed_state: TimedVehicleState, lane: DgLanelet, gen: int) -> Successors:
        if self.params.solve:
            return self.get_successors_solve(timed_state=timed_state, lane=lane, gen=gen)
        else:
            return self.get_successors_approx(timed_state=timed_state, lane=lane, gen=gen)

    def get_acc_dst(self, state: VehicleState, gen: int) -> Tuple[Set[float], Set[float]]:
        u0 = VehicleCommands(acc=0.0, ddelta=0.0)
        cond_gen = gen < self.params.max_gen
        dst_vals = self._bicycle_dyn.u_dst if cond_gen else {0.0}
        acc_vals = self._bicycle_dyn.get_feasible_acc(x=state, dt=self.params.dt, u0=u0)
        #todo: was: if not_cond_gen: remove all acc<0 and set them to acc=0.0 -> only move forward.
        # issue: this cond_gen is never reached. We should use the condition that the goal is reached, if any.
        # if not cond_gen:
        #     for acc in list(acc_vals):  # todo [LEON]: issue when acceleration values is =0!
        #         if acc <= 0.0:  # todo [LEON]: added <= instead of < -> fix this
        #             acc_vals.remove(acc)
        #             # acc_vals.add(0.0) # todo [LEON]: removed temporarily.
        return acc_vals, dst_vals

    def get_successors_approx(self, timed_state: TimedVehicleState, lane: DgLanelet, gen: int) -> Successors:
        """
        Approximate method to grow trajectory tree (fast)
        Predicts progress along reference using curvature
        Samples discrete grid of progress (from acceleration) and deviation
        Steers car using kinematic model to reach close to target point
        """

        dt = float(self.params.dt)
        # l = self.params.vg.l
        # l = self.params.vg.length # todo lf or lr, /2?
        l = self.params.vg.lr

        # Calculate initial pose
        start_arr = np.array([timed_state[1].x, timed_state[1].y])
        th_start = timed_state[1].theta
        # n_i is distance from lane
        # mui is relative angle between closest pose on lane and state
        # along_i is progress along lane of closest point to state
        along_i, n_i, mui = self.get_curv(state=timed_state[1], lane=lane)

        # Calculate real axle translation and rotation
        offset_0, offset_i = np.array([0, 0]), np.array([-l, 0])
        p_i, th_i = self._get_target(lane=lane, progress=along_i, offset_target=offset_0)
        q_start = geo.SE2_from_translation_angle(t=start_arr, theta=th_start)
        p_start = SE2_apply_T2(q_start, offset_i)

        def get_progress(acc: float, K: float) -> float:
            """Progress along reference using curvature"""
            vf = timed_state[1].vx + acc * dt

            # we don't allow velocities smaller than 0 (no trajectories moving backwards)
            if vf < 0.0:
                vf = 0.0

            return (vf * dt) / (1 - n_i * K)

        def get_corrected_distance(acc: float) -> float:
            """Progress along reference iteratively corrected using curvature"""
            curv = 0.0
            # if acc == 0.0:  # todo [LEON]: workaround for now. Issue when acc==0.0 (division by zero happens in line 218)
            #     acc = 0.01
            dist = get_progress(acc=acc, K=curv)
            # assert dist != 0, "Progress can't be zero.  Choose another acceleration."
            # if dist is 0.0, no progress is done with this acceleration and initial velocity.
            tol = 1e-3
            if dist > tol:
                for i in range(5):
                    p_f, th_f = self._get_target(lane=lane, progress=along_i + dist, offset_target=offset_0)
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
        acc_vals, dst_vals = self.get_acc_dst(state=timed_state[1], gen=gen)
        successors: Successors = {}

        # Sample progress using acceleration
        for accel in acc_vals:
            distance = get_corrected_distance(acc=accel)
            n_scale = distance if self.params.dst_scale else 1.0

            # Sample deviation as a function of dst
            for dst in dst_vals:
                if distance > 0.0:
                    # Calculate target pose of rear axle
                    nf = self.params.n_factor * n_i + dst * n_scale
                    offset_t = np.array([-l, nf])
                    p_t, th_t = self._get_target(lane=lane, progress=along_i + distance, offset_target=offset_t)

                    # Steer from initial to final position using kinematic model
                    #  No slip at rear axle assumption --> Rear axle moves along a circle
                    dlb_t = p_t - p_start
                    Lb_t = np.linalg.norm(dlb_t)
                    alpb = math.atan2(dlb_t[1], dlb_t[0]) - th_start
                    tan_st = 4 * math.sin(alpb) * l / Lb_t
                    st_f = min(max(math.atan(tan_st), -st_max), st_max)
                    dst_f = min(max((st_f - timed_state[1].delta) / dt, -dst_max), dst_max)
                elif distance == 0.0:
                    # meaning that there is no change from the previous state
                    accel = 0
                    dst_f = 0.0
                else:
                    assert False, "Something went wrong here."

                # Propagate inputs to obtain exact final state
                u = VehicleCommands(acc=accel, ddelta=dst_f)
                state_f, states_t = self.get_successor(state=timed_state, u=u)
                successors[u] = (state_f, states_t)

        return successors

    # todo still need to change fro states to timed states here
    def get_successors_solve(self, timed_state: TimedVehicleState, lane: DgLanelet, gen: int) -> Successors:
        """
        Accurate method to grow trajectory tree (slow)
        Samples discrete grid of velocity (from acceleration) and deviation
        Solves a two point boundary value problem to calculate steering angle
        Propagates states using calculated steering and kinematic model
        """
        state = timed_state[1]
        dt = float(self.params.dt)
        s_init, n_init, mui = self.get_curv(state=state, lane=lane)
        successors: Dict[VehicleCommands, Tuple[VehicleState, List[VehicleState]]] = {}

        # Steering rate bounds
        dst_max = self.params.dst_max
        lb = max(-dst_max, (-self.params.st_max - state.delta) / dt)
        ub = min(+dst_max, (+self.params.st_max - state.delta) / dt)
        acc_vals, dst_vals = self.get_acc_dst(state=state, gen=gen)

        def equation_forward(vars_in, acc: float) -> Tuple[float, float]:
            """Euler forward integration (cartesian) to obtain curvilinear state"""
            u = VehicleCommands(acc=acc, ddelta=vars_in[0])
            state_end, _ = self.get_successor(state=timed_state, u=u, samp=False)
            _, n, mu = self.get_curv(state=state_end[1], lane=lane)
            return n, mu

        def equation_min(vars_in, acc: float, nfinal: float) -> float:
            """Function for optimiser"""
            n, mu = equation_forward(vars_in, acc=acc)
            return (n - nfinal) ** 2 + float(np.abs(mu) > np.pi / 2) * 10000

        def get_dst_guess() -> float:
            """Initial guess for optimisation, obtained from target yaw rate"""
            p_t, th_t = self._get_target(lane=lane, progress=s_init + distance, offset_target=np.array([0, 0]))
            d_ang = th_t - state.theta
            while d_ang > +np.pi:
                d_ang -= 2 * np.pi
            while d_ang < -np.pi:
                d_ang += 2 * np.pi
            # dst_i = (math.atan(d_ang * 2 * self.params.vg.l / state.v * dt) - state.st) / dt
            dst_i = (math.atan(d_ang * 2 * self.params.vg.lr / state.vx * dt) - state.delta) / dt
            dst_i = min(max(dst_i, lb), ub)
            return dst_i

        # Sample velocities
        for accel in acc_vals:
            vf = state.vx + accel * dt
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
                    print(f"Opt failed: {state}, acc={accel}, nf={nf}")
                    continue

                # Propagate inputs to obtain final state
                u_f = VehicleCommands(acc=accel, ddelta=dst_f)
                state_f, states_t = self.get_successor(state=timed_state, u=u_f)
                successors[u_f] = (state_f, states_t)
        return successors

