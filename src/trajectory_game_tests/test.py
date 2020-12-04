from typing import Dict
from networkx import MultiDiGraph

from trajectory_game import AllTrajectories, Trajectory, RuleEvaluationResult

from .trajectory_generation import generate_trajectory_graph, trajectory_graph_to_list
from .trajectory_scoring import score_trajectories, remove_dominated_trajectories


def test_trajectory_selection():

    # Change trajectory graph params in trajectory_generation
    G: MultiDiGraph = generate_trajectory_graph()
    all_traj: AllTrajectories = trajectory_graph_to_list(G=G)

    # Change world params in trajectory_scoring
    result: Dict[Trajectory, Dict[str, RuleEvaluationResult]]
    result = score_trajectories(all_traj=all_traj)
    result_nondom = remove_dominated_trajectories(result=result)
    a = 2
