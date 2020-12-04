from typing import List, Set, Dict
from networkx import MultiDiGraph, topological_sort

from .structures import VehicleState
from .transitions import Trajectory
from .world import World
from .rules import RuleEvaluationResult
from .metrics import evaluate_rules


class AllTrajectories:
    ego_name: str
    all_trajectories: List[Trajectory]
    expanded: Set[VehicleState] = []

    def __init__(self, G: MultiDiGraph, ego_name: str):
        self.ego_name = ego_name
        self.all_trajectories = []
        self.expanded = set()
        for n1 in topological_sort(G):
            traj: List[VehicleState] = [n1]
            self.expand_graph(G=G, node=n1, traj=traj)

    def expand_graph(self, G: MultiDiGraph, node: VehicleState, traj: List[VehicleState]):
        if node in self.expanded:
            return
        succ = list(G.successors(node))
        if not succ:
            self.all_trajectories.append(Trajectory(traj))
        else:
            for n2 in succ:
                traj1: List[VehicleState] = traj + [n2]
                self.expand_graph(G=G, node=n2, traj=traj1)
        self.expanded.add(node)

    def evaluate_trajectories(self, world: World) -> Dict[Trajectory, Dict[str, RuleEvaluationResult]]:
        ret = {}
        for traj in self.all_trajectories:
            result = evaluate_rules(trajectory=traj, world=world, ego_name=self.ego_name)
            ret[traj] = result
        return ret

    def __iter__(self):
        return self.all_trajectories.__iter__()
