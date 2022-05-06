import os
from typing import Set, Mapping, FrozenSet, Sequence, List

from geometry import translation_angle_from_SE2

from dg_commons import PlayerName
from dg_commons.maps import DgLanelet
from dg_commons.planning import TrajectoryGraph, RefLaneGoal, Trajectory
from dg_commons.sim.models import CAR
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.scenarios import load_commonroad_scenario, DgScenario
from dg_commons_dev.utils import get_project_root_dir
from trajectory_games import PosetalPreference, TrajectoryGenParams, TrajectoryGenerator, TrajectoryWorld
from commonroad_challenge.stochastic_decision_making.utils import UncertainActionEstimator
from decimal import Decimal as D
from possibilities import ProbDist
from fractions import Fraction

P1 = PlayerName("p1")
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
        P1: [RefLaneGoal(ref_lane=dglanelet_1, goal_progress=0.8)],
        P2: [RefLaneGoal(ref_lane=dglanelet_2, goal_progress=0.8)],
        P3: [RefLaneGoal(ref_lane=dglanelet_3, goal_progress=0.8)],
    }

    geos = {
        P1: VehicleGeometry.default_car(),
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
        P1: VehicleState(x=x1[0][0], y=x1[0][1], theta=x1[1], vx=5., delta=0),
        P2: VehicleState(x=x2[0][0], y=x2[0][1], theta=x2[1], vx=5., delta=0),
        P3: VehicleState(x=x3[0][0], y=x3[0][1], theta=x3[1], vx=5., delta=0)
    }

    return dgscenario, geos, goals, initial_states


def get_default_cr_preferences(n_other_agents: int) -> Mapping[PlayerName, PosetalPreference]:
    pref_structures: Mapping[PlayerName, PosetalPreference] = {}
    default_str = "default_commonroad"
    pref_structures[PlayerName("Ego")] = PosetalPreference(pref_str="default_commonroad_ego", use_cache=False)
    for n in range(n_other_agents):
        pname = PlayerName("P" + str(n + 1))
        pref_structures[pname] = PosetalPreference(pref_str=default_str, use_cache=False)
    return pref_structures


def get_traj_gen_params():
    u_acc = frozenset([-3.0, -2.0])
    u_dst = frozenset([0.0])

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
        max_gen=5,
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


if __name__ == "__main__":
    # STEPS
    # 1) generate actions for all players
    # 2) Generate preferences for all players
    # 3) Evaluate metrics on all joint trajectories (intelligently)
    # 4) By knowing preference of others, keep "n" best ones and rank them
    # 5) Compute probability of each action from previous ranking
    # 6) Propagate uncertainty in outcome space
    # 7) Ego picks his action with the uncertainty in the outcome space

    # 2)
    # prefs = get_default_cr_preferences(2)
    # a =10

    dgscenario, geos, goals, initial_states = get_scenario_and_all()

    traj_gen_params = get_traj_gen_params()
    actions = generate_actions(initial_states=initial_states, ref_lane_goals=goals, traj_gen_params=traj_gen_params)
    world: TrajectoryWorld = TrajectoryWorld(map_name=dgscenario.scenario.scenario_id.map_name, scenario=dgscenario,
                                             geo=geos, goals=goals)
    estimator = UncertainActionEstimator(world=world, actions=actions)
    estimator.compute_player_probabilities(player_name=P1)
