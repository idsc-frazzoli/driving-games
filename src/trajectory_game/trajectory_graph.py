from typing import List, Set, Dict
from networkx import MultiDiGraph, topological_sort

from .structures import VehicleState
from .paths import Trajectory
from .world import World
from .metrics_def import MetricEvaluationResult
from .metrics import evaluate_metrics


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

    def evaluate_trajectories(self, world: World) -> Dict[Trajectory, Dict[str, MetricEvaluationResult]]:
        ret = {}
        for traj in self.all_trajectories:
            result = evaluate_metrics(trajectory=traj, world=world, ego_name=self.ego_name)
            ret[traj] = result
        return ret

    def __iter__(self):
        return self.all_trajectories.__iter__()
