import os
from typing import Mapping

from dg_commons import PlayerName
from dg_commons.planning import Trajectory, RefLaneGoal
from dg_commons.sim.scenarios import DgScenario, load_commonroad_scenario
from dg_commons_dev.utils import get_project_root_dir
from driving_games.metrics_structures import MetricEvaluationContext


def get_defualt_evaluation_context() -> MetricEvaluationContext:
    SCENARIOS_DIR = os.path.join(get_project_root_dir(), "scenarios")
    scenario, _ = load_commonroad_scenario("DEU_Ffb-1_7_T-1", SCENARIOS_DIR)
    dgscenario = DgScenario(scenario)

    # todo create some fake joint trajectories and goals

    joint_trajectories: Mapping[PlayerName, Trajectory] = {}
    goals: Mapping[PlayerName, RefLaneGoal] = {}

    return MetricEvaluationContext(dgscenario=dgscenario, trajectories=joint_trajectories, goals=goals)


def test_metrics_1():
    evaluation_context = get_defualt_evaluation_context()
    # todo


def test_metrics_2():
    evaluation_context = get_defualt_evaluation_context()
    # todo
