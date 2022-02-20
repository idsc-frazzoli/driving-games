import os
from copy import copy
from dataclasses import replace
from decimal import Decimal as D
from typing import Dict, Mapping, Callable

import numpy as np

from dg_commons import fd, fs, PlayerName
from dg_commons.sim.models import kmh2ms
from dg_commons.sim.models.vehicle_ligths import NO_LIGHTS
from dg_commons.sim.scenarios import load_commonroad_scenario
from dg_commons.sim.scenarios.agent_from_commonroad import dglane_from_position
from dg_commons_dev.utils import get_project_root_dir
from games import GameSpec, UncertaintyParams
from possibilities import PossibilityDist, PossibilitySet
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
)

P1 = PlayerName("P1")
P2 = PlayerName("P2")
P3 = PlayerName("P3")
P4 = PlayerName("P4")
P5 = PlayerName("P5")
P6 = PlayerName("P6")
P7 = PlayerName("P7")
P8 = PlayerName("P8")

SCENARIOS_DIR = os.path.join(get_project_root_dir(), "scenarios")

simple_intersection, _ = load_commonroad_scenario("DEU_Ffb-1_7_T-1", SCENARIOS_DIR)
s_lane1 = dglane_from_position(np.array([0, 0]), simple_intersection.lanelet_network, succ_lane_selection=1)
s_lane2 = dglane_from_position(np.array([70, -14]), simple_intersection.lanelet_network, succ_lane_selection=0)
s_lane3 = dglane_from_position(np.array([85, 8]), simple_intersection.lanelet_network, succ_lane_selection=0)

param_2p = DgSimpleParams(
    track_dynamics_param=dyn_p0,
    shared_resources_ds=D("1.5"),
    col_check_dt=D("0.76"),
    ref_lanes={P1: s_lane1, P2: s_lane2},
    scenario=simple_intersection,
    progress={P1: (D(140), D(165)), P2: (D(180), D(200))},
    plot_limits=[[40, 100], [-25, 25]],
    min_safety_distance=6,
)

param_3p = replace(
    param_2p,
    ref_lanes={P1: s_lane1, P2: s_lane2, P3: s_lane3},
    progress={P1: (D(140), D(165)), P2: (D(178), D(200)), P3: (D(115), D(140))},
)

uncertainty_sets = UncertaintyParams(poss_monad=PossibilitySet(), mpref_builder=SetWorstCasePreference)
uncertainty_prob = UncertaintyParams(poss_monad=PossibilityDist(), mpref_builder=ProbPrefExpectedValue)


def get_simple_int_2p_sets() -> GameSpec:
    desc = """
    Plain 4way intersection. 2 players. Set-based uncertainty.
    """
    return GameSpec(desc, get_driving_game(param_2p, uncertainty_sets))


def get_simple_int_3p_sets() -> GameSpec:
    desc = """
    Plain 4way intersection. 3 players. Set-based uncertainty.
    """
    return GameSpec(desc, get_driving_game(param_3p, uncertainty_sets))


def get_simple_int_2p_prob() -> GameSpec:
    desc = """
    Plain 4way intersection. 2 players.
    Probability-based uncertainty (expected value).
    """
    return GameSpec(desc, get_driving_game(copy(param_2p), uncertainty_prob))


def get_simple_int_3p_prob() -> GameSpec:
    desc = """
    Plain 4way intersection. 3 players.
    Probability-based uncertainty (expected value).
    """
    return GameSpec(desc, get_driving_game(copy(param_3p), uncertainty_prob))


multilane_intersection, _ = load_commonroad_scenario("USA_Lanker-1_1_T-1.xml", SCENARIOS_DIR)
mint_lane1 = dglane_from_position(np.array([0, 25]), multilane_intersection.lanelet_network, succ_lane_selection=0)
mint_lane2 = dglane_from_position(np.array([-8, -2]), multilane_intersection.lanelet_network, succ_lane_selection=0)
mint_lane3 = dglane_from_position(np.array([20, 10]), multilane_intersection.lanelet_network, succ_lane_selection=0)
mint_lane4 = dglane_from_position(np.array([9, 28]), multilane_intersection.lanelet_network, succ_lane_selection=0)
mint_lane5 = dglane_from_position(np.array([-15, 13]), multilane_intersection.lanelet_network, succ_lane_selection=0)
mint_lane6 = dglane_from_position(np.array([5, 25]), multilane_intersection.lanelet_network, succ_lane_selection=0)

mint_param_2p = DgSimpleParams(
    track_dynamics_param=dyn_p0,
    shared_resources_ds=D(1.5),
    col_check_dt=D("0.76"),
    ref_lanes={P1: mint_lane1, P2: mint_lane2},
    scenario=multilane_intersection,
    progress={P1: (D(10), D(30)), P2: (D(10), D(35))},
    plot_limits=[[-30, 30], [-12, 35]],
    min_safety_distance=6,
)
mint_param_3p = DgSimpleParams(
    track_dynamics_param=dyn_p0,
    shared_resources_ds=D(1.5),
    col_check_dt=D("0.76"),
    ref_lanes={P1: mint_lane1, P2: mint_lane2, P3: mint_lane3},
    scenario=multilane_intersection,
    progress={P1: (D(10), D(30)), P2: (D(10), D(35)), P3: (D(15), D(40))},
    plot_limits=[[-30, 30], [-12, 35]],
    min_safety_distance=6,
)
mint_param_4p = DgSimpleParams(
    track_dynamics_param=dyn_p0,
    shared_resources_ds=D(1.5),
    col_check_dt=D("0.76"),
    ref_lanes={P1: mint_lane1, P2: mint_lane2, P3: mint_lane3, P4: mint_lane4},
    scenario=multilane_intersection,
    progress={P1: (D(10), D(30)), P2: (D(10), D(35)), P3: (D(15), D(40)), P4: (D(15), D(40))},
    # progress={P1: (D(10), D(35)), P2: (D(5), D(35)), P3: (D(10), D(40)), P4: (D(10), D(40))}, # node with no eq among 1,2,3
    plot_limits=[[-30, 30], [-12, 35]],
    min_safety_distance=6,
)
mint_param_5p = replace(
    mint_param_4p,
    ref_lanes={P1: mint_lane1, P2: mint_lane2, P3: mint_lane3, P4: mint_lane4, P5: mint_lane5},
    progress={P1: (D(10), D(30)), P2: (D(10), D(35)), P3: (D(15), D(40)), P4: (D(15), D(40)), P5: (D(0), D(20))},
)
mint_param_6p = replace(
    mint_param_4p,
    ref_lanes={P1: mint_lane1, P2: mint_lane2, P3: mint_lane3, P4: mint_lane4, P5: mint_lane5, P6: mint_lane6},
    progress={
        P1: (D(10), D(30)),
        P2: (D(10), D(35)),
        P3: (D(15), D(40)),
        P4: (D(15), D(40)),
        P5: (D(0), D(20)),
        P6: (D(10), D(30)),
    },
)


def get_multilane_int_2p_sets() -> GameSpec:
    desc = """
    Multilane intersection modeled after USA_Lanker-1_1_T-1. xx players. Set-based uncertainty.
    """
    return GameSpec(desc, get_driving_game(mint_param_2p, uncertainty_sets))


def get_multilane_int_3p_sets() -> GameSpec:
    desc = """
    Multilane intersection modeled after USA_Lanker-1_1_T-1. xx players. Set-based uncertainty.
    """
    return GameSpec(desc, get_driving_game(mint_param_3p, uncertainty_sets))


def get_multilane_int_4p_sets() -> GameSpec:
    desc = """
    Multilane intersection modeled after USA_Lanker-1_1_T-1. xx players. Set-based uncertainty.
    """
    return GameSpec(desc, get_driving_game(mint_param_4p, uncertainty_sets))


def get_multilane_int_5p_sets() -> GameSpec:
    desc = """
    Multilane intersection modeled after USA_Lanker-1_1_T-1. xx players. Set-based uncertainty.
    """
    return GameSpec(desc, get_driving_game(mint_param_5p, uncertainty_sets))


def get_multilane_int_6p_sets() -> GameSpec:
    desc = """
    Multilane intersection modeled after USA_Lanker-1_1_T-1. xx players. Set-based uncertainty.
    """
    return GameSpec(desc, get_driving_game(mint_param_6p, uncertainty_sets))


complex_intersection, _ = load_commonroad_scenario("DEU_Muc-1_1_T-1", SCENARIOS_DIR)

c_lane1 = dglane_from_position(np.array([-19, 0]), complex_intersection.lanelet_network, succ_lane_selection=0)
c_lane2 = dglane_from_position(np.array([10, -14]), complex_intersection.lanelet_network, succ_lane_selection=0)
c_lane3 = dglane_from_position(np.array([15, -10]), complex_intersection.lanelet_network, succ_lane_selection=0)
c_lane4 = dglane_from_position(np.array([-10, -12]), complex_intersection.lanelet_network, succ_lane_selection=0)
c_lane5 = dglane_from_position(
    np.array([-10, -17]), complex_intersection.lanelet_network, init_lane_selection=1, succ_lane_selection=0
)
c_lane6 = dglane_from_position(np.array([30, 9]), complex_intersection.lanelet_network, succ_lane_selection=1)

c_param_6p = DgSimpleParams(
    track_dynamics_param=dyn_p0,
    shared_resources_ds=D(1.5),
    col_check_dt=D("0.51"),
    ref_lanes={P1: c_lane1, P2: c_lane2, P3: c_lane3, P4: c_lane4, P5: c_lane5, P6: c_lane6},
    scenario=complex_intersection,
    progress={
        P1: (D(30), D(60)),
        P2: (D(10), D(40)),
        P3: (D(10), D(50)),
        P4: (D(30), D(70)),
        P5: (D(25), D(50)),
        P6: (D(20), D(50)),
    },
    plot_limits=[[-50, 50], [-50, 50]],
    min_safety_distance=6,
)


def get_complex_int_6p_sets() -> GameSpec:
    desc = """
    Complex intersection modeled after DEU_Muc-1_1_T-1. xx players. Set-based uncertainty.
    """
    return GameSpec(desc, get_driving_game(c_param_6p, uncertainty_sets))


def get_complex_int_xxp_sets() -> GameSpec:
    desc = """
    Complex intersection modeled after DEU_Muc-1_1_T-1. xx players. Set-based uncertainty.
    """
    return GameSpec(desc, get_driving_game(c_param_6p, uncertainty_sets))


# made into a callable to avoid long import times
driving_games_zoo: Mapping[str, Callable[[], GameSpec]] = fd(
    {
        "simple_int_2p_sets": get_simple_int_2p_sets,
        "simple_int_3p_sets": get_simple_int_3p_sets,
        "simple_int_2p_prob": get_simple_int_2p_prob,
        "simple_int_3p_prob": get_simple_int_3p_prob,
        "multilane_int_2p_sets": get_multilane_int_2p_sets,
        "multilane_int_3p_sets": get_multilane_int_3p_sets,
        "multilane_int_4p_sets": get_multilane_int_4p_sets,
        "multilane_int_5p_sets": get_multilane_int_5p_sets,
        "multilane_int_6p_sets": get_multilane_int_6p_sets,
        "complex_int_6p_sets": get_complex_int_6p_sets,
        "complex_int_xxp_sets": get_complex_int_xxp_sets,
    }
)

games_zoo: Dict[str, Callable[[], GameSpec]] = {}

games_zoo.update(driving_games_zoo)
