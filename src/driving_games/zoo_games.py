import os
from copy import copy
from decimal import Decimal as D
from typing import Dict, Mapping

import numpy as np

from dg_commons import fd, fs, PlayerName
from dg_commons.sim.models import kmh2ms
from dg_commons.sim.models.vehicle_ligths import NO_LIGHTS
from dg_commons.sim.scenarios import load_commonroad_scenario
from dg_commons.sim.scenarios.agent_from_commonroad import dglane_from_position
from dg_commons_dev.utils import get_project_root_dir
from games import GameSpec, UncertaintyParams
from possibilities import PossibilitySet, PossibilityDist
from preferences import SetWorstCasePreference
from preferences.preferences_probability import ProbPrefExpectedValue
from . import VehicleTrackDynamicsParams
from .dg_def import DgSimpleParams
from .dg_factory import get_driving_game

dyn_p0 = VehicleTrackDynamicsParams(
    max_speed=D(kmh2ms(50)),
    min_speed=D(0),
    available_accels=fs({D(-1), D(0), D(1)}),
    max_wait=D(0),
    lights_commands=fs({NO_LIGHTS}),
    shared_resources_ds=1.5,
)

P1 = PlayerName("P1")
P2 = PlayerName("P2")
P3 = PlayerName("P3")
P4 = PlayerName("P4")

SCENARIOS_DIR = os.path.join(get_project_root_dir(), "scenarios")
# complex_intersection, _ = load_commonroad_scenario("DEU_Muc-1_1_T-1", SCENARIOS_DIR)
# c_lane1 = dglane_from_position(np.array([0, 0]), complex_intersection.lanelet_network)
# c_lane2 = dglane_from_position(np.array([5, 5]), complex_intersection.lanelet_network)
# fixme complex intersection needs to be intialized properly

simple_intersection, _ = load_commonroad_scenario("DEU_Ffb-1_7_T-1", SCENARIOS_DIR)
s_lane1 = dglane_from_position(np.array([0, 0]), simple_intersection.lanelet_network, succ_lane_selection=1)
s_lane2 = dglane_from_position(np.array([70, -14]), simple_intersection.lanelet_network, succ_lane_selection=0)
s_lane3 = dglane_from_position(np.array([85, 8]), simple_intersection.lanelet_network, succ_lane_selection=0)

param_2p = DgSimpleParams(
    track_dynamics_param=dyn_p0,
    shared_resources_ds=D(1),
    col_check_dt=D("0.51"),
    ref_lanes={P1: s_lane1, P2: s_lane2},
    scenario=simple_intersection,
    progress={P1: (D(135), D(160)), P2: (D(180), D(195))},
    plot_limits=[[40, 100], [-25, 25]],
    min_safety_distance=4,
)

param_3p = DgSimpleParams(
    track_dynamics_param=dyn_p0,
    shared_resources_ds=D(1),
    col_check_dt=D("0.51"),
    ref_lanes={P1: s_lane1, P2: s_lane2, P3: s_lane3},
    scenario=simple_intersection,
    progress={P1: (D(135), D(160)), P2: (D(180), D(195)), P3: (D(120), D(140))},
    plot_limits=[[40, 100], [-25, 25]],
    min_safety_distance=4,
)

uncertainty_sets = UncertaintyParams(poss_monad=PossibilitySet(), mpref_builder=SetWorstCasePreference)
uncertainty_prob = UncertaintyParams(poss_monad=PossibilityDist(), mpref_builder=ProbPrefExpectedValue)


def get_4way_int_2p_sets() -> GameSpec:
    desc = """
    Plain 4way intersection. 2 players. Set-based uncertainty.
    """
    return GameSpec(desc, get_driving_game(param_2p, uncertainty_sets))


def get_4way_int_3p_sets() -> GameSpec:
    desc = """
    Plain 4way intersection. 3 players. Set-based uncertainty.
    """
    return GameSpec(desc, get_driving_game(param_3p, uncertainty_sets))


def get_4way_int_2p_prob() -> GameSpec:
    desc = """
    Plain 4way intersection. 2 players.
    Probability-based uncertainty (expected value).
    """
    return GameSpec(desc, get_driving_game(copy(param_2p), uncertainty_prob))


def get_4way_int_3p_prob() -> GameSpec:
    desc = """
    Plain 4way intersection. 3 players.
    Probability-based uncertainty (expected value).
    """
    return GameSpec(desc, get_driving_game(copy(param_3p), uncertainty_prob))


driving_games_zoo: Mapping[str, GameSpec] = fd(
    {
        "4way_int_2p_sets": get_4way_int_2p_sets(),
        "4way_int_3p_sets": get_4way_int_3p_sets(),
        "4way_int_2p_prob": get_4way_int_2p_prob(),
        "4way_int_3p_prob": get_4way_int_3p_prob(),
    }
)

games_zoo: Dict[str, GameSpec] = {}

games_zoo.update(driving_games_zoo)
