import os
from functools import partial
from time import perf_counter
from typing import Dict, Set

from yaml import safe_load

from games import PlayerName, MonadicPreferenceBuilder
from possibilities import PossibilitySet
from preferences import SetPreference1

from .game_def import EXP_ACCOMP, JOIN_ACCOMP
from .config import config_dir
from .structures import VehicleGeometry, VehicleState, TrajectoryParams
from .trajectory_generator import TransitionGenerator
from .metrics import MetricEvaluation
from .preference import PosetalPreference
from .trajectory_game import TrajectoryGame, TrajectoryGamePlayer, LeaderFollowerGame, LeaderFollowerParams
from .trajectory_world import TrajectoryWorld
from .visualization import TrajGameVisualization
from _tmp._deprecated.world import load_driving_game_map, LaneSegmentHashable, get_lane_from_node_sequence

__all__ = [
    "get_trajectory_game",
    "get_leader_follower_game",
]

players_file = os.path.join(config_dir, "players.yaml")
lanes_file = os.path.join(config_dir, "lanes.yaml")
leader_follower_file = os.path.join(config_dir, "leader_follower.yaml")
with open(players_file) as load_file:
    config = safe_load(load_file)
with open(lanes_file) as load_file:
    config_lanes = safe_load(load_file)[config["map_name"]]
with open(leader_follower_file) as load_file:
    config_lf = safe_load(load_file)["leader_follower"]


def get_trajectory_game(config_str: str = "basic") -> TrajectoryGame:
    tic = perf_counter()
    lanes: Dict[PlayerName, Set[LaneSegmentHashable]] = {}
    geometries: Dict[PlayerName, VehicleGeometry] = {}
    players: Dict[PlayerName, TrajectoryGamePlayer] = {}
    duckie_map = load_driving_game_map(config["map_name"])

    ps = PossibilitySet()
    mpref_build: MonadicPreferenceBuilder = SetPreference1

    for pname, pconfig in config[config_str]["players"].items():
        lanes[pname] = set()
        # TODO[SIR]: Check that all lanes start at the same node
        for lane_id in pconfig["lane"]:
            lane = config_lanes[lane_id]
            lane_seg = get_lane_from_node_sequence(m=duckie_map, node_sequence=lane)
            lanes[pname].add(LaneSegmentHashable.initializor(lane_seg))
        geometries[pname] = VehicleGeometry.from_config(pconfig["vg"])
        param = TrajectoryParams.from_config(name=pconfig["traj"], vg_name=pconfig["vg"])
        traj_gen = TransitionGenerator(params=param)
        pref = PosetalPreference(pref_str=pconfig["pref"], use_cache=False)
        state = VehicleState.from_config(name=pconfig["state"], lane=next(iter(lanes[pname])))
        players[pname] = TrajectoryGamePlayer(
            name=pname,
            state=ps.unit(state),
            actions_generator=traj_gen,
            preference=pref,
            monadic_preference_builder=mpref_build,
            vg=geometries[pname],
        )

    world = TrajectoryWorld(map_name=config["map_name"], geo=geometries, lanes=lanes)
    get_outcomes = partial(MetricEvaluation.evaluate, world=world)
    game = TrajectoryGame(
        world=world,
        game_players=players,
        ps=ps,
        get_outcomes=get_outcomes,
        game_vis=TrajGameVisualization(world=world),
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

    lf = LeaderFollowerParams(leader=PlayerName(config_lf["leader"]),
                              follower=PlayerName(config_lf["follower"]),
                              pref_leader=pref_leader,
                              prefs_follower_est=prefs_follower_est,
                              pref_follower_real=get_pref1(name=config_lf["pref_follower_real"]),
                              antichain_comparison=ac_comp[ac_cfg],
                              solve_time=float(config_lf["solve_time"]),
                              simulation_step=float(config_lf["simulation_step"]),
                              terminal_progress=config_lf["terminal_progress"],
                              update_prefs=config_lf["update_prefs"])

    # Init pref dict with the correct order of follower prefs from list instead of set
    all_prefs: Dict[PlayerName, Set[PosetalPreference]] = {
        lf.leader: {pref_leader},
        lf.follower: set(prefs_follower)
    }
    game.game_vis.init_pref_dict(values=all_prefs)
    pref_dict_f = game.game_vis.get_pref_dict(player=lf.follower)
    for p_f in prefs_follower:
        _ = pref_dict_f[p_f]

    game_lf = LeaderFollowerGame(world=game.world, game_players=game.game_players,
                                 ps=game.ps, get_outcomes=game.get_outcomes,
                                 game_vis=game.game_vis, lf=lf)
    return game_lf
