import math
from abc import ABC, abstractmethod
from time import perf_counter
from typing import FrozenSet, Set, List, Dict, Tuple
import numpy as np
from decimal import Decimal as D

from duckietown_world import LaneSegment, SE2Transform
from duckietown_world.utils import SE2_apply_R2
from networkx import MultiDiGraph, topological_sort

from games import PlayerName
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
        if (player, state) in self._cache:
            return self._cache[(player, state)]
        assert world is not None
        tic = perf_counter()
        G = self._get_trajectory_graph(state=state, lane=world.get_lane(player=player))
        if isinstance(graph, MultiDiGraph):
            graph.__init__(G)
        trajectories = self._trajectory_graph_to_list(G=G)
        toc = perf_counter() - tic
        print(f"Trajectory generation time = {toc:.2f} s")
        ret = frozenset(trajectories)
        self._cache[(player, state)] = ret
        return ret

    def _get_trajectory_graph(self, state: VehicleState, lane: LaneSegment) -> MultiDiGraph:
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
            u_mean = self.get_mean_actions(state=s1, lane=lane)
            successors = self._bicycle_dyn.successors(x=s1, dt=self.params.dt, u0=u_mean)
            for u, s2 in successors.items():
                if s2 not in G.nodes:
                    add_node(s2, gen=n_gen + 1)
                    if n_gen + 1 < self.params.max_gen:
                        stack.append(s2)
                G.add_edge(s1, s2, u=u, gen=n_gen)

        return G

    def get_mean_actions(self, state: VehicleState, lane: LaneSegment) -> VehicleActions:

        scale = 1.5         # Pure pursuit tuning parameter
        dt = self.params.dt

        # Calculate intial pose of car
        se2 = SE2Transform(p=np.array([state.x, state.y]), theta=float(state.th))
        start = lane.lane_pose_from_SE2Transform(qt=se2)

        # Pure pursuit lookahead point
        distance_vx = float(state.v * dt)
        progress_end = start.along_lane + distance_vx * scale
        beta_end = lane.beta_from_along_lane(along_lane=progress_end)
        end_value = lane.center_point(beta=beta_end)
        offset = np.array([0, 0])
        end = SE2_apply_R2(end_value, offset)

        # Pure pursuit controller
        start_arr = np.array([float(_) for _ in [state.x, state.y]])
        dl = end - start_arr
        L = np.linalg.norm(dl)
        l = float(self.params.vg.l)
        sa_cog = math.atan(math.tan(float(state.st))/2.0)
        alp  = math.atan2(dl[1], dl[0]) - (float(state.th) + sa_cog)

        # Steering rate from required yawrate - using RK2 integrator
        dst  = (D(math.atan((8/scale)*math.sin(alp )*l/L - math.tan(float(state.st)))) - state.st) / dt
        return VehicleActions(acc=D("0"), dst=dst)

    @staticmethod
    def _trajectory_graph_to_list(G: MultiDiGraph) -> Set[Trajectory]:
        expanded = set()
        all_traj: Set[Trajectory] = set()
        for n1 in topological_sort(G):
            traj: List[VehicleState] = [n1]
            TrajectoryGenerator1._expand_graph(G=G, node=n1, traj=traj, all_traj=all_traj, expanded=expanded)
        return all_traj

    @staticmethod
    def _expand_graph(
        G: MultiDiGraph,
        node: VehicleState,
        traj: List[VehicleState],
        all_traj: Set[Trajectory],
        expanded: Set[VehicleState],
    ):
        if node in expanded:
            return
        successors = list(G.successors(node))
        if not successors:
            all_traj.add(Trajectory(traj))
        else:
            for n2 in successors:
                traj1: List[VehicleState] = traj + [n2]
                TrajectoryGenerator1._expand_graph(
                    G=G, node=n2, traj=traj1, all_traj=all_traj, expanded=expanded
                )
        expanded.add(node)
