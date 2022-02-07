import os
from copy import copy
from dataclasses import replace
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
from preferences import SetPreference1
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
    shared_resources_ds=D(1.5),
)

P1 = PlayerName("P1")
P2 = PlayerName("P2")
SCENARIOS_DIR = os.path.join(get_project_root_dir(), "scenarios")

# complex_intersection, _ = load_commonroad_scenario("DEU_Muc-1_1_T-1", SCENARIOS_DIR)
# c_lane1 = dglane_from_position(np.array([0, 0]), complex_intersection.lanelet_network)
# c_lane2 = dglane_from_position(np.array([5, 5]), complex_intersection.lanelet_network)
# fixme complex intersection needs to be intialized properly

simple_intersection, _ = load_commonroad_scenario("DEU_Ffb-1_7_T-1", SCENARIOS_DIR)
s_lane1 = dglane_from_position(np.array([0, 0]), simple_intersection.lanelet_network, succ_lane_selection=1)
s_lane2 = dglane_from_position(np.array([70, -14]), simple_intersection.lanelet_network, succ_lane_selection=0)

p0 = DgSimpleParams(
    track_dynamics_param=dyn_p0,
    shared_resources_ds=D(0),
    col_check_dt=D("0.51"),
    ref_lanes={P1: s_lane1, P2: s_lane2},
    scenario=simple_intersection,
    progress={P1: (D(135), D(160)), P2: (D(175), D(190))},
    plot_limits=[[40, 100], [-25, 25]],
    min_safety_distance=4,
)

uncertainty_sets = UncertaintyParams(poss_monad=PossibilitySet(), mpref_builder=SetPreference1)
uncertainty_prob = UncertaintyParams(poss_monad=PossibilityDist(), mpref_builder=ProbPrefExpectedValue)

p_asym = replace(p0, progress={P1: (D(140), D(160)), P2: (D(175), D(190))})


def get_sym() -> GameSpec:
    desc = """
    Simple intersection. (Super) symmetric case. Min v = 1. Set-based uncertainty.
    """
    return GameSpec(desc, get_driving_game(p0, uncertainty_sets))


def get_asym() -> GameSpec:
    desc = """
    Slightly asymmetric case. Min v = 1. Set-based uncertainty.
    """
    return GameSpec(desc, get_driving_game(p_asym, uncertainty_sets))


def get_sym_prob() -> GameSpec:
    desc = """
    Super symmetric case. Min v = 1.
    Probability-based uncertainty (expected value).
    """
    return GameSpec(desc, get_driving_game(copy(p0), uncertainty_prob))


def get_asym_prob() -> GameSpec:
    desc = """
    Slightly asymmetric case. Probability-based uncertainty (expected value).
    """
    return GameSpec(desc, get_driving_game(copy(p_asym), uncertainty_prob))


driving_games_zoo: Mapping[str, GameSpec] = fd(
    {
        "sym_v1_sets": get_sym(),
        "asym_v1_sets": get_asym(),
        "sym_v1_prob": get_sym_prob(),
        "asym_v1_prob": get_asym_prob(),
    }
)

games_zoo: Dict[str, GameSpec] = {}

games_zoo.update(driving_games_zoo)
