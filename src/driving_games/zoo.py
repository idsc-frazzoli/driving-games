from dataclasses import replace
from decimal import Decimal as D
from typing import Dict

from games import GameSpec
from possibilities import PossibilitySet, ProbabilityFraction
from preferences import SetPreference1
from preferences.preferences_probability import ProbPrefExpectedValue
from .game_generation import get_two_vehicle_game, TwoVehicleSimpleParams, TwoVehicleUncertaintyParams
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
uncertainty_sets = TwoVehicleUncertaintyParams(poss_monad=PossibilitySet(), mpref_builder=SetPreference1)
uncertainty_prob = TwoVehicleUncertaintyParams(
    poss_monad=ProbabilityFraction(), mpref_builder=ProbPrefExpectedValue
)
p_sym = p0


def get_sym() -> GameSpec:
    desc = """
    Super symmetric case. Min v = 1. Set-based uncertainty.
    """
    return GameSpec(desc, get_two_vehicle_game(p_sym, uncertainty_sets))


p_asym = replace(p0, road_lane_offset=D(4))  # to the right


def get_asym() -> GameSpec:
    desc = """
    Slightly asymmetric case. West is advantaged.  
    Min v = 1. Set-based uncertainty.
    """
    return GameSpec(desc, get_two_vehicle_game(p_asym, uncertainty_sets))


p_asym_minv0 = replace(p_asym, min_speed=D(0))


def get_asym_minv0() -> GameSpec:
    desc = """
    Slightly asymmetric case. West is advantaged.  
    Min v = 0. Set-based uncertainty. 
    """
    return GameSpec(desc, get_two_vehicle_game(p_asym_minv0, uncertainty_sets))


def get_sym_prob() -> GameSpec:
    desc = """
    Super symmetric case. Min v = 1. 
    Probability-based uncertainty (expected value).
    """
    return GameSpec(desc, get_two_vehicle_game(p_sym, uncertainty_prob))


def get_asym_prob() -> GameSpec:
    desc = """
    Slightly asymmetric case. West is advantaged.  
    Min v = 1. Probability-based uncertainty (expected value).
    """
    return GameSpec(desc, get_two_vehicle_game(p_asym, uncertainty_prob))


def get_asym_minv0_prob() -> GameSpec:
    desc = """
    Slightly asymmetric case. West is advantaged.  
    Min v = 0. Probability-based uncertainty (expected value).
    """
    return GameSpec(desc, get_two_vehicle_game(p_asym_minv0, uncertainty_prob))


driving_games_zoo: Dict[str, GameSpec] = {
    "sym_v1_sets": get_sym(),
    "asym_v1_sets": get_asym(),
    "asym_v0_sets": get_asym_minv0(),
    "sym_v1_prob": get_sym_prob(),
    "asym_v1_prob": get_asym_prob(),
    "asym_v0_prob": get_asym_minv0_prob(),
}
