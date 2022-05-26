import copy
import os
from decimal import Decimal as D
from fractions import Fraction
from functools import partial
from random import random
from time import perf_counter
from typing import Mapping, FrozenSet, List

from frozendict import frozendict
from geometry import translation_angle_from_SE2

import trajectory_games.metrics
from dg_commons import PlayerName, iterate_dict_combinations
from dg_commons.maps import DgLanelet
from dg_commons.planning import RefLaneGoal, Trajectory, JointTrajectories
from dg_commons.sim.models import CAR
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.scenarios import load_commonroad_scenario, DgScenario
from dg_commons_dev.utils import get_project_root_dir
from driving_games.metrics_structures import JointPlayerOutcome, EvaluatedMetric
from possibilities import ProbDist
from trajectory_games.stochastic_decision_making.uncertain_preferences import UncertainPreferenceOutcomes

from trajectory_games import PosetalPreference, TrajectoryGenParams, TrajectoryGenerator, TrajectoryWorld, \
    MetricEvaluation

EGO = PlayerName("Ego")
P2 = PlayerName("p2")
P3 = PlayerName("p3")


def get_scenario_and_all():
    SCENARIOS_DIR = os.path.join(get_project_root_dir(), "scenarios")
    scenario, _ = load_commonroad_scenario("DEU_Ffb-1_7_T-1", SCENARIOS_DIR)
    dgscenario = DgScenario(scenario)

    # defines reference lanelets for players
    lanelet_network = scenario.lanelet_network

    points_from_first = 4
    points_from_last = 4

    lane_11 = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49570))
    lane_12 = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49598))
    lane_13 = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49576))

    lane_21 = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49574))
    lane_22 = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49600))
    lane_23 = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49566))

    lane_31 = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49564))
    lane_32 = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49586))
    lane_33 = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49568))

    dglane1_ctrl_points = (
            lane_11.control_points[-points_from_last:-1]
            + lane_12.control_points
            + lane_13.control_points[1:points_from_first]
    )

    dglane2_ctrl_points = (
            lane_21.control_points[-points_from_last:-1]
            + lane_22.control_points
            + lane_23.control_points[1:points_from_first]
    )

    dglane3_ctrl_points = (
            lane_31.control_points[-points_from_last:-1]
            + lane_32.control_points
            + lane_33.control_points[1:points_from_first]
    )

    dglanelet_1 = DgLanelet(dglane1_ctrl_points)
    dglanelet_2 = DgLanelet(dglane2_ctrl_points)
    dglanelet_3 = DgLanelet(dglane3_ctrl_points)

    goals = {
        EGO: [RefLaneGoal(ref_lane=dglanelet_1, goal_progress=0.8)],
        P2: [RefLaneGoal(ref_lane=dglanelet_2, goal_progress=0.8)],
        P3: [RefLaneGoal(ref_lane=dglanelet_3, goal_progress=0.8)],
    }

    geos = {
        EGO: VehicleGeometry.default_car(),
        P2: VehicleGeometry.default_car(),
        P3: VehicleGeometry.default_car(),
    }

    # Define trajectories for players
    x_1_translation_angles = [
        translation_angle_from_SE2(dglanelet_1.center_point(beta)) for beta in range(len(dglanelet_1.control_points))
    ]
    x_2_translation_angles = [
        translation_angle_from_SE2(dglanelet_2.center_point(beta)) for beta in range(len(dglanelet_2.control_points))
    ]
    x_3_translation_angles = [
        translation_angle_from_SE2(dglanelet_3.center_point(beta)) for beta in range(len(dglanelet_3.control_points))
    ]

    x1 = x_1_translation_angles[0]
    x2 = x_2_translation_angles[0]
    x3 = x_3_translation_angles[0]

    initial_states = {
        EGO: VehicleState(x=x1[0][0], y=x1[0][1], theta=x1[1], vx=5., delta=0),
        P2: VehicleState(x=x2[0][0], y=x2[0][1], theta=x2[1], vx=5., delta=0),
        P3: VehicleState(x=x3[0][0], y=x3[0][1], theta=x3[1], vx=5., delta=0)
    }

    return dgscenario, geos, goals, initial_states


def get_preferences() -> Mapping[PlayerName, PosetalPreference]:
    pref_structures: Mapping[PlayerName, PosetalPreference] = {}

    pref_structures[EGO] = PosetalPreference(pref_str="only_squared_acc", use_cache=False)
    pref_structures[P2] = PosetalPreference(pref_str="only_squared_acc", use_cache=False)
    pref_structures[P3] = PosetalPreference(pref_str="only_squared_acc", use_cache=False)

    return pref_structures


def get_traj_gen_params():
    u_acc = frozenset([-3.0, -2.0])
    u_dst = frozenset([-0.4])

    vg = VehicleGeometry(
        vehicle_type=CAR,
        m=1500.0,
        Iz=1300,
        w_half=1.8,
        lf=2.0,
        lr=2.0,
        c_drag=0.3756,
        a_drag=2,
        e=0.5,
        color="royalblue",
    )

    params = TrajectoryGenParams(
        solve=False,
        s_final=-1,  # todo: adapt metrics to use this
        # s_final=-1,
        max_gen=4,
        dt=D("1.0"),
        # keep at max 1 sec, increase k_maxgen in trajectory_generator for having more generations
        u_acc=u_acc,
        u_dst=u_dst,
        v_max=15.0,
        v_min=0.0,
        st_max=0.9,
        dst_max=0.4,
        dt_samp=D("0.1"),
        dst_scale=False,
        n_factor=1.0,
        vg=vg,
        acc_max=10.0,
        v_switch=4.0
    )

    return params


def generate_actions(initial_states: Mapping[PlayerName, VehicleState],
                     ref_lane_goals: Mapping[PlayerName, List[RefLaneGoal]],
                     traj_gen_params: TrajectoryGenParams):
    trajs: Mapping[PlayerName, FrozenSet[Trajectory]] = {}
    for player_name, initial_state in initial_states.items():
        p_traj_gen = TrajectoryGenerator(params=traj_gen_params, ref_lane_goals=ref_lane_goals[player_name])
        trajs[player_name] = p_traj_gen.get_actions(state=initial_state, return_graphs=False)

    return trajs


def test_equivalent_outcomes():
    # test that if all joint trajectories have same outcomes, all trajectories will be equivalent for Ego
    dgscenario, geos, goals, initial_states = get_scenario_and_all()

    traj_gen_params = get_traj_gen_params()
    all_actions = generate_actions(initial_states=initial_states, ref_lane_goals=goals, traj_gen_params=traj_gen_params)

    joint_a_o: Mapping[JointTrajectories, JointPlayerOutcome] = {}

    from random import randint

    metrics = trajectory_games.metrics.get_metrics_set()
    random_numbers = {}
    for metric in metrics:
        random_numbers[metric] = randint(0, 1)

    for joint_traj in set(iterate_dict_combinations(all_actions)):
        players_outcome = {}
        for player in joint_traj.keys():
            dummy_outcome = {}
            for metric in metrics:
                dummy_outcome[metric] = EvaluatedMetric(name=metric.get_name(), value=random_numbers[metric])
            players_outcome[player] = frozendict(dummy_outcome)
        joint_a_o[joint_traj] = copy.deepcopy(frozendict(players_outcome))

    only_acc_pref = PosetalPreference(pref_str="only_squared_acc", use_cache=False)

    pref_leon_dev_2 = PosetalPreference(pref_str="pref_leon_dev_2", use_cache=False)
    pref_leon_dev_3 = PosetalPreference(pref_str="pref_leon_dev_3", use_cache=False)

    uncertain_pref_distr = {
        P2: ProbDist({pref_leon_dev_2: Fraction(1, 3), pref_leon_dev_3: Fraction(2, 3)}),
        P3: ProbDist({pref_leon_dev_2: Fraction(3, 4), pref_leon_dev_3: Fraction(1, 4)})
    }
    upo = UncertainPreferenceOutcomes(my_name=EGO, pref_distr=uncertain_pref_distr,
                                      joint_actions_outcomes_mapping=joint_a_o, ego_pref=only_acc_pref)

    outcome_distr = upo.outcome_distr()
    selected_actions = upo.action_selector(method="avg")

    assert set(outcome_distr.keys()) == all_actions[PlayerName("Ego")]
    assert selected_actions == all_actions[PlayerName("Ego")]


def test_2():
    dgscenario, geos, goals, initial_states = get_scenario_and_all()

    traj_gen_params = get_traj_gen_params()
    preferences = get_preferences()
    all_actions = generate_actions(initial_states=initial_states, ref_lane_goals=goals, traj_gen_params=traj_gen_params)
    world: TrajectoryWorld = TrajectoryWorld(map_name=dgscenario.scenario.scenario_id.map_name, scenario=dgscenario,
                                             geo=geos, goals=goals)
    get_outcomes = partial(MetricEvaluation.evaluate, world=world)

    joint_a_o: Mapping[JointTrajectories, JointPlayerOutcome] = {}

    for joint_traj in set(iterate_dict_combinations(all_actions)):
        joint_a_o[joint_traj] = get_outcomes(joint_traj)

    only_acc_pref = PosetalPreference(pref_str="only_squared_acc", use_cache=False)

    pref_leon_dev_2 = PosetalPreference(pref_str="pref_leon_dev_2", use_cache=False)
    pref_leon_dev_3 = PosetalPreference(pref_str="pref_leon_dev_3", use_cache=False)

    uncertain_pref_distr = {
        P2: ProbDist({pref_leon_dev_2: Fraction(1, 3), pref_leon_dev_3: Fraction(2, 3)}),
        P3: ProbDist({pref_leon_dev_2: Fraction(3, 4), pref_leon_dev_3: Fraction(1, 4)})
    }
    upo = UncertainPreferenceOutcomes(my_name=EGO, pref_distr=uncertain_pref_distr,
                                      joint_actions_outcomes_mapping=joint_a_o, ego_pref=only_acc_pref)

    outcome_distr = upo.outcome_distr()
    selected_actions = upo.action_selector(method="avg")


if __name__ == "__main__":
    test_equivalent_outcomes()
    # test_2()
