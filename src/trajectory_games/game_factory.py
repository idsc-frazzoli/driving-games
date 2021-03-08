import os
from time import perf_counter
from typing import Dict
from yaml import safe_load

from games import PlayerName, MonadicPreferenceBuilder
from possibilities import PossibilitySet
from preferences import SetPreference1

from .config import config_dir
from .structures import VehicleGeometry, VehicleState, TrajectoryParams
from .trajectory_generator import TrajectoryGenerator1
from .metrics import evaluate_metrics
from .preference import PosetalPreference
from .trajectory_game import TrajectoryGame, TrajectoryGamePlayer
from .trajectory_world import TrajectoryWorld
from .visualization import TrajGameVisualization
from world import load_driving_game_map, LaneSegmentHashable, get_lane_from_node_sequence

__all__ = [
    "get_trajectory_game",
]


def get_trajectory_game() -> TrajectoryGame:

    tic = perf_counter()
    players_file = os.path.join(config_dir, "players.yaml")
    lanes_file = os.path.join(config_dir, "lanes.yaml")
    with open(players_file) as load_file:
        config = safe_load(load_file)
    with open(lanes_file) as load_file:
        config_lanes = safe_load(load_file)[config["map_name"]]
    lanes: Dict[PlayerName, LaneSegmentHashable] = {}
    geometries: Dict[PlayerName, VehicleGeometry] = {}
    players: Dict[PlayerName, TrajectoryGamePlayer] = {}
    duckie_map = load_driving_game_map(config["map_name"])

    ps = PossibilitySet()
    mpref_build: MonadicPreferenceBuilder = SetPreference1

    for pname, pconfig in config["players"].items():
        lane_seg = get_lane_from_node_sequence(m=duckie_map, node_sequence=config_lanes[pconfig["lane"]])
        lanes[pname] = LaneSegmentHashable.initializor(lane_seg)
        geometries[pname] = VehicleGeometry.from_config(pconfig["vg"])
        param = TrajectoryParams.from_config(name=pconfig["traj"], vg_name=pconfig["vg"])
        traj_gen = TrajectoryGenerator1(params=param)
        pref = PosetalPreference(pref_str=pconfig["pref"])
        state = VehicleState.from_config(name=pconfig["state"], lane=lanes[pname])
        players[pname] = TrajectoryGamePlayer(
            name=pname,
            state=ps.unit(state),
            actions_generator=traj_gen,
            preference=pref,
            monadic_preference_builder=mpref_build,
            vg=geometries[pname],
        )

    world = TrajectoryWorld(map_name=config["map_name"], geo=geometries, lanes=lanes)
    game = TrajectoryGame(
        world=world,
        game_players=players,
        ps=ps,
        get_outcomes=evaluate_metrics,
        game_vis=TrajGameVisualization(world=world),
    )
    toc = perf_counter() - tic
    print(f"Game creation time = {toc:.2f} s")
    return game
