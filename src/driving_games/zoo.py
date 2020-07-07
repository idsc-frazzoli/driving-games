from dataclasses import replace
from decimal import Decimal as D
from typing import Dict

from belief_games import get_leader_follower_game, get_alone_game
from games import GameSpec
from .game_generation import get_two_vehicle_game, TwoVehicleSimpleParams
from .structures import NO_LIGHTS

road = D(6)
p0 = TwoVehicleSimpleParams(
    side=D(8),
    road=road,
    road_lane_offset=road / 2,  # center
    max_speed=D(5),
    min_speed=D(1),
    max_wait=D(1),
    available_accels=frozenset({D(-2), D(-1), D(0), D(+1)}),
    collision_threshold=3.0,
    light_actions=frozenset({NO_LIGHTS}),
    dt=D(1),
    first_progress=D(0),
    second_progress=D(0),
    shared_resources_ds=D(1.5),
)

p_sym = p0


def get_sym() -> GameSpec:
    desc = """
    Super symmetric case. Min v = 1.
    """
    return GameSpec(desc, get_two_vehicle_game(p_sym))


p_asym = replace(p0, road_lane_offset=D(4))  # to the right


def get_asym() -> GameSpec:
    desc = """
    Slightly asymmetric case. West is advantaged.  
    Min v = 1.
    """
    return GameSpec(desc, get_two_vehicle_game(p_asym))


p_asym_minv0 = replace(p_asym, min_speed=D(0))


def get_asym_minv0() -> GameSpec:
    desc = """
    Slightly asymmetric case. West is advantaged.  
    Min v = 0.
    """
    return GameSpec(desc, get_two_vehicle_game(p_asym_minv0))


def get_asym_lf() -> GameSpec:
    desc = """
    Asymmetric "Leader Follower" game where one player does not think the other is there.
    """
    return GameSpec(desc, get_leader_follower_game(p_asym_minv0))


def get_alone() -> GameSpec:
    desc = """
    Alone Game (only one player)
    """
    return GameSpec(desc, get_alone_game(p_asym_minv0))


driving_games_zoo: Dict[str, GameSpec] = {}

driving_games_zoo["sym_v1"] = get_sym()
driving_games_zoo["asym_v1"] = get_asym()
driving_games_zoo["asym_v0"] = get_asym_minv0()

driving_games_zoo["lf_v0"] = get_asym_lf()
driving_games_zoo["alone"] = get_alone()