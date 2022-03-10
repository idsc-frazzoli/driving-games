import os
from functools import partial
from time import perf_counter
from typing import Dict, Set, Tuple, Optional, List

import numpy as np
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad.scenario.scenario import Scenario
from dg_commons.sim.scenarios import DgScenario
from dg_commons.planning import RefLaneGoal
from shapely.geometry import Polygon
from yaml import safe_load
from crash import logger

from dg_commons import PlayerName, SE2Transform
from dg_commons.maps import DgLanelet
from games import MonadicPreferenceBuilder
from possibilities import PossibilitySet
from preferences import SetPreference
from dg_commons.sim.scenarios import load_commonroad_scenario
from .config.ral import config_dir_ral
from .game_def import EXP_ACCOMP, JOIN_ACCOMP
from .metrics import MetricEvaluation
from .preference import PosetalPreference
from .structures import TrajectoryParams
from dg_commons.sim.models.vehicle import VehicleState, VehicleGeometry
from .trajectory_game import TrajectoryGame, TrajectoryGamePlayer, LeaderFollowerGame, LeaderFollowerParams
from .trajectory_generator import TransitionGenerator
from .trajectory_world import TrajectoryWorld
from .visualization import TrajGameVisualization
from .config import CONFIG_DIR

__all__ = ["get_trajectory_game", "get_leader_follower_game", "get_simple_traj_game_leon"]

players_file = os.path.join(config_dir_ral, "players.yaml")
# leader_follower_file = os.path.join(config_dir, "leader_follower.yaml")
with open(players_file) as load_file:
    config = safe_load(load_file)


# with open(leader_follower_file) as load_file:
#     config_lf = safe_load(load_file)["leader_follower"]

# todo: move this somewhere appropriate
def from_config(name: str) -> "VehicleState":
    filename = os.path.join(CONFIG_DIR, "initial_states.yaml")
    with open(filename) as load_file:
        config = safe_load(load_file)

    if name in config.keys():
        pconfig = config[name]
        state = VehicleState(
            x=pconfig["x0"], y=pconfig["y0"], theta=pconfig["th0"], vx=pconfig["v0"], delta=pconfig["st0"]
        )
    else:
        print(f"Failed to initialise VehicleState from {name}")
        state = None
    return state


def get_goal_polygon(lanelet: DgLanelet, goal: np.ndarray) -> Polygon:
    beta, _ = lanelet.find_along_lane_closest_point(p=goal)
    progress = lanelet.along_lane_from_beta(beta)
    points: List[np.ndarray] = []

    for dx, dy in [(-1, -1), (-1, 1), (1, 1), (1, -1)]:
        s_f, n_f = progress + dy * config["goal"]["long_dist"], dx * config["goal"]["lat_dist"]
        xy_f, _ = TransitionGenerator._get_target(lane=lanelet, progress=s_f, offset_target=np.array([0, n_f]))
        points.append(xy_f)
    return Polygon(points)


def lanelet_from_ids(lanelet_ids: List[int], network: LaneletNetwork):
    lanelets = []
    merged_lanelet = network.find_lanelet_by_id(lanelet_ids[0])
    for enum, current_id in enumerate(lanelet_ids):
        if enum == 0:
            continue
        current_lanelet = network.find_lanelet_by_id(current_id)
        merged_lanelet = Lanelet.merge_lanelets(lanelet1=merged_lanelet, lanelet2=current_lanelet)
    return merged_lanelet


def get_lanelet_from_points(start: np.ndarray, goal: np.ndarray, lanelet_network: LaneletNetwork):
    start_id = lanelet_network.find_lanelet_by_position([start])[0][0]
    goal_id = lanelet_network.find_lanelet_by_position([goal])[0][0]
    assert start_id, "No lanelet found for the start position"
    assert goal_id, "No lanelet found for the goal position"

    start_lanelet = lanelet_network.find_lanelet_by_id(start_id)
    all_lanes = Lanelet.all_lanelets_by_merging_successors_from_lanelet(
        lanelet=start_lanelet, network=lanelet_network, max_length=2000
    )

    merged_lanes, merged_lane_ids = all_lanes

    ref_lanes: List[DgLanelet] = []
    for enum, merged_lane_id in enumerate(merged_lane_ids):
        if goal_id in merged_lane_id:
            ref_lanes.append(DgLanelet.from_commonroad_lanelet(merged_lanes[enum]))
        else:
            continue
    return ref_lanes


def get_simple_traj_game_leon(config_str: str) -> TrajectoryGame:
    tic = perf_counter()
    geometries: Dict[PlayerName, VehicleGeometry] = {}
    players: Dict[PlayerName, TrajectoryGamePlayer] = {}
    goals: Dict[PlayerName, List[RefLaneGoal]] = {}
    scenario: DgScenario
    logger.info(f"Loading Scenario: {config['map_name']}", end=" ...")
    scenarios_dir = "/home/leon/Documents/repos/driving-games/scenarios"
    scenario, _ = load_commonroad_scenario(config["map_name"], scenarios_dir=scenarios_dir)
    logger.info("Done.")

    required_goal_progress: float = 0.8

    ps = PossibilitySet()
    mpref_build: MonadicPreferenceBuilder = SetPreference

    for pname, pconfig in config[config_str]["players"].items():
        if pname == "Ambulance": #todo [LEON] Figure out why ambulance does not work
             continue
        logger.info(f"Extracting lanes: {pname}", end=" ...")

        # extract data from config file
        state = from_config(name=pconfig["state"])
        state_init = np.array([state.x, state.y])
        p_goals = [np.array(goal) for goal in pconfig["goals"]]
        player_color = pconfig["vg"].replace("car_", "")
        pref = PosetalPreference(pref_str=pconfig["pref"], use_cache=False)

        # transition generator
        ref_lanes = get_lanelet_from_points(start=state_init, goal=p_goals[0], lanelet_network=scenario.lanelet_network)
        ref_lane_goals = [RefLaneGoal(ref_lane=lane, goal_progress=required_goal_progress) for lane in ref_lanes]
        trans_param = TrajectoryParams.from_config(name=pconfig["traj"], vg_name=player_color)
        traj_gen = TransitionGenerator(params=trans_param, ref_lane_goals=ref_lane_goals)

        goals[pname] = ref_lane_goals

        geometries[pname] = VehicleGeometry.default_car(color=player_color)

        players[pname] = TrajectoryGamePlayer(
            name=pname,
            state=ps.unit(state),
            actions_generator=traj_gen,
            preference=pref,
            monadic_preference_builder=mpref_build,
            vg=geometries[pname],
        )

    world = TrajectoryWorld(map_name=config["map_name"], scenario=scenario, geo=geometries, goals=goals)
    get_outcomes = partial(MetricEvaluation.evaluate, world=world)
    game = TrajectoryGame(
        world=world,
        game_players=players,
        ps=ps,
        get_outcomes=get_outcomes,
        game_vis=TrajGameVisualization(world=world, plot_limits=config["plot_limits"]),
    )

    toc = perf_counter() - tic
    logger.info(f"Game creation time = {toc:.2f} s")

    return game


def get_trajectory_game(config_str: str = "basic") -> TrajectoryGame:
    tic = perf_counter()
    lanes: Dict[PlayerName, List[Tuple[DgLanelet, Optional[Polygon]]]] = {}
    geometries: Dict[PlayerName, VehicleGeometry] = {}
    players: Dict[PlayerName, TrajectoryGamePlayer] = {}
    scenario: Scenario
    print(f"Loading Scenario: {config['map_name']}", end=" ...")
    scenarios_dir = "/home/leon/Documents/repos/driving-games/scenarios"
    scenario, _ = load_commonroad_scenario(config["map_name"], scenarios_dir=scenarios_dir)
    print("Done")
    lane_network: LaneletNetwork = scenario.lanelet_network

    ps = PossibilitySet()
    mpref_build: MonadicPreferenceBuilder = SetPreference

    for pname, pconfig in config[config_str]["players"].items():
        print(f"Extracting lanes: {pname}", end=" ...")
        state = VehicleState.from_config(name=pconfig["state"])
        state_init = np.array([state.x, state.y])

        if pconfig["goals"] is None:
            lanes_all = DgLanelet.from_start(
                lane_network=lane_network, start=state_init, max_length=config["goal"]["max_length"]
            )
            lanes[pname] = list((lane, None) for lane in lanes_all)
        else:
            goals = list(np.array(goal) for goal in pconfig["goals"])
            lanes_all = DgLanelet.from_ends(lane_network=lane_network, start=state_init, goals=goals)
            lanes[pname] = []
            for lane, goal in zip(lanes_all, goals):
                lanes[pname].append((lane, get_goal_polygon(lanelet=lane, goal=goal)))

        # Reset pose to the center of the reference lane
        if len(lanes_all) == 0:
            raise ValueError(f"No lanes for the existing point: {state}")
        _, q = lanes_all[0].find_along_lane_closest_point(p=state_init)
        se2_init = SE2Transform.from_SE2(q)
        state.x, state.y, state.th = se2_init.p[0], se2_init.p[1], se2_init.theta

        geometries[pname] = VehicleGeometry.from_config(pconfig["vg"])
        param = TrajectoryParams.from_config(name=pconfig["traj"], vg_name=pconfig["vg"])
        traj_gen = TransitionGenerator(params=param)
        pref = PosetalPreference(pref_str=pconfig["pref"], use_cache=False)
        players[pname] = TrajectoryGamePlayer(
            name=pname,
            state=ps.unit(state),
            actions_generator=traj_gen,
            preference=pref,
            monadic_preference_builder=mpref_build,
            vg=geometries[pname],
        )

    world = TrajectoryWorld(map_name=config["map_name"], scenario=scenario, geo=geometries, lanes=lanes)
    get_outcomes = partial(MetricEvaluation.evaluate, world=world)
    game = TrajectoryGame(
        world=world,
        game_players=players,
        ps=ps,
        get_outcomes=get_outcomes,
        game_vis=TrajGameVisualization(world=world, plot_limits=config["plot_limits"]),
    )
    toc = perf_counter() - tic
    print(f"Game creation time = {toc:.2f} s")
    return game


ac_comp = {"JOIN_ACCOMP": JOIN_ACCOMP, "EXP_ACCOMP": EXP_ACCOMP}


def get_leader_follower_game() -> LeaderFollowerGame:
    game = get_trajectory_game(config_str="lf")

    def get_pref1(name: str) -> PosetalPreference:
        return PosetalPreference(pref_str=name, use_cache=False)

    ac_cfg = config_lf["antichain_comparison"]
    if ac_cfg not in ac_comp:
        raise ValueError(f"ac_comp - {ac_cfg} not in {ac_comp.keys()}")

    pref_leader = get_pref1(name=config_lf["pref_leader"])
    prefs_follower = [get_pref1(name=p) for p in config_lf["prefs_follower_est"]]
    prefs_follower_est = game.ps.lift_many(prefs_follower)

    lf = LeaderFollowerParams(
        leader=PlayerName(config_lf["leader"]),
        follower=PlayerName(config_lf["follower"]),
        pref_leader=pref_leader,
        prefs_follower_est=prefs_follower_est,
        pref_follower_real=get_pref1(name=config_lf["pref_follower_real"]),
        antichain_comparison=ac_comp[ac_cfg],
        solve_time=float(config_lf["solve_time"]),
        simulation_step=float(config_lf["simulation_step"]),
        update_prefs=config_lf["update_prefs"],
    )

    # Init pref dict with the correct order of follower prefs from list instead of set
    all_prefs: Dict[PlayerName, Set[PosetalPreference]] = {lf.leader: {pref_leader}, lf.follower: set(prefs_follower)}
    game.game_vis.init_pref_dict(values=all_prefs)
    pref_dict_f = game.game_vis.get_pref_dict(player=lf.follower)
    for p_f in prefs_follower:
        _ = pref_dict_f[p_f]

    game_lf = LeaderFollowerGame(
        world=game.world,
        game_players=game.game_players,
        ps=game.ps,
        get_outcomes=game.get_outcomes,
        game_vis=game.game_vis,
        lf=lf,
    )
    return game_lf
