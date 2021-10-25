from dataclasses import replace
from decimal import Decimal as D
from typing import Dict, Mapping

import numpy as np

from dg_commons import fd, fs, PlayerName
from dg_commons.sim.models import kmh2ms
from dg_commons.sim.scenarios import load_commonroad_scenario
from dg_commons.sim.scenarios.agent_from_commonroad import dglane_from_position
from games import GameSpec, UncertaintyParams
from possibilities import PossibilitySet, PossibilityDist
from preferences import SetPreference1
from preferences.preferences_probability import ProbPrefExpectedValue
from . import VehicleTrackDynamicsParams
from .game_generation import get_two_vehicle_game, DGSimpleParams
from .structures import NO_LIGHTS

dyn_p0 = VehicleTrackDynamicsParams(
    max_speed=D(kmh2ms(50)),
    min_speed=D(0),
    available_accels=fs({D(-1), D(0), D(1)}),
    max_wait=D(0),
    lights_commands=fs({NO_LIGHTS}),
    shared_resources_ds=D(1.5),
)

P1 = PlayerName("P1")
P2 = PlayerName("P2")

complex_intersection, _ = load_commonroad_scenario("DEU_Muc-1_1_T-1")
c_lane1 = dglane_from_position(np.array([0, 0]), complex_intersection.lanelet_network)
c_lane2 = dglane_from_position(np.array([5, 5]), complex_intersection.lanelet_network)

simple_intersection, _ = load_commonroad_scenario("ITA_Segrate-1_3_T-1")

p0 = DGSimpleParams(
    track_dynamics_param=dyn_p0,
    shared_resources_ds=D(0),
    game_dt=D(1),
    ref_lanes={P1: c_lane1, P2: c_lane2},
    scenario=simple_intersection,
    progress={P1: (D(0), D(8)), P2: (D(0), D(8))},
)

uncertainty_sets = UncertaintyParams(poss_monad=PossibilitySet(), mpref_builder=SetPreference1)
uncertainty_prob = UncertaintyParams(poss_monad=PossibilityDist(), mpref_builder=ProbPrefExpectedValue)
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


driving_games_zoo: Mapping[str, GameSpec] = fd(
    {
        "sym_v1_sets": get_sym(),
        "asym_v1_sets": get_asym(),
        "asym_v0_sets": get_asym_minv0(),
        "sym_v1_prob": get_sym_prob(),
        "asym_v1_prob": get_asym_prob(),
        "asym_v0_prob": get_asym_minv0_prob(),
    }
)

games_zoo: Dict[str, GameSpec] = {}

games_zoo.update(driving_games_zoo)
