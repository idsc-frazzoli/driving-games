from abc import ABC, abstractmethod
from time import perf_counter
from typing import FrozenSet, Set, List

from networkx import MultiDiGraph, topological_sort

from .structures import VehicleState, TrajectoryParams
from .static_game import ActionSetGenerator
from .paths import Trajectory
from .trajectory_world import TrajectoryWorld
from .bicycle_dynamics import BicycleDynamics

__all__ = ["TrajectoryGenerator", "TrajectoryGenerator1"]


class TrajectoryGenerator(ActionSetGenerator[VehicleState, Trajectory, TrajectoryWorld], ABC):
    @abstractmethod
    def get_action_set(self, state: VehicleState, world: TrajectoryWorld, **kwargs) -> FrozenSet[Trajectory]:
        pass


class TrajectoryGenerator1(TrajectoryGenerator):
    def __init__(self, params: TrajectoryParams):
        self.params = params
        self._bicycle_dyn = BicycleDynamics(params=params)

    def get_action_set(self, state: VehicleState, world: TrajectoryWorld, graph: MultiDiGraph = None) -> FrozenSet[Trajectory]:
        G = self._get_trajectory_graph(state=state)
        if isinstance(graph, MultiDiGraph):
            graph.__init__(G)
        trajectories = self._trajectory_graph_to_list(G=G)
        return frozenset(trajectories)

    def _get_trajectory_graph(self, state: VehicleState) -> MultiDiGraph:
        stack = list([state])
        G = MultiDiGraph()

        def add_node(s, gen):
            G.add_node(s, gen=gen, x=s.x, y=s.y)

        add_node(state, gen=0)
        i: int = 0
        expanded = set()
        tic = perf_counter()
        while stack:
            i += 1
            s1 = stack.pop(0)
            assert s1 in G.nodes
            if s1 in expanded:
                continue
            n_gen = G.nodes[s1]["gen"]
            expanded.add(s1)
            successors = self._bicycle_dyn.successors(s1, self.params.dt)
            for u, s2 in successors.items():
                if s2 not in G.nodes:
                    add_node(s2, gen=n_gen + 1)
                    if n_gen + 1 < self.params.max_gen:
                        stack.append(s2)
                G.add_edge(s1, s2, u=u, gen=n_gen)

        toc = perf_counter() - tic
        print("Trajectory graph generation time = {} s".format(toc))
        return G

    @staticmethod
    def _trajectory_graph_to_list(G: MultiDiGraph) -> Set[Trajectory]:
        tic = perf_counter()
        expanded = set()
        all_traj: Set[Trajectory] = set()
        for n1 in topological_sort(G):
            traj: List[VehicleState] = [n1]
            TrajectoryGenerator1._expand_graph(G=G, node=n1, traj=traj, all_traj=all_traj, expanded=expanded)
        toc = perf_counter() - tic
        print("Trajectory list generation time = {} s".format(toc))
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
