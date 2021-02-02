from decimal import Decimal as D, localcontext
from typing import Dict
import os

import duckietown_world as dw
from duckietown_world.world_duckietown.duckiebot import DB18

from games import PlayerName, UncertaintyParams
from possibilities import PossibilitySet, PossibilityDist
from preferences import SetPreference1
from preferences.preferences_probability import ProbPrefExpectedValue

from driving_games.structures import NO_LIGHTS

from duckie_games.utils import get_lane_segments, merge_lanes, Lane, load_duckie_map_from_yaml
from duckie_games.game_generation import DuckieGameParams
from duckie_games.structures import DuckieGeometry

__all__ = ['two_player_duckie_game_parameters', 'two_player_duckie_game_parameters_stretched', 'uncertainty_sets', 'uncertainty_prob']

module_path = os.path.dirname(__file__)

"""
Collection of different parameters for a Duckiegame
"""

# Scaled version of p_sym() of driving_games
with localcontext() as ctx:
    ctx.prec = 2
    map_name = '4way'
    duckie_map = dw.load_map(map_name)
    player_nb = 2
    duckie_names = [PlayerName("Duckie_1"), PlayerName("Duckie_2")]
    lane_names = {
        duckie_names[0]: ['ls051', 'ls033', 'ls016'],
        # duckie_names[0]: ['ls026', 'ls022', 'L13'],
        # duckie_names[1]: ['ls041', 'ls036', 'ls026'],
        duckie_names[1]: ['ls041', 'ls035', 'ls050']

    }
    lanes: Dict[PlayerName, Lane]
    lanes = {dn: merge_lanes(get_lane_segments(duckie_map=duckie_map, lane_names=lane_names[dn])) for dn in duckie_names}

    duck_g = DuckieGeometry(
            mass=D(1000),
            width=D(DB18().width),
            length=D(DB18().length),
            color=(1, 0, 0),
            height=D(DB18().height)
    )

    duckie_geometries = {dn: duck_g for dn in duckie_names}

    max_paths = {dn: D(lanes[dn].get_lane_length()) * D(1) for dn in duckie_names}
    max_path_driving_games = D(21)
    max_path_first = max_paths[duckie_names[0]]

    # scale the parameters
    max_speed = D(5) * max_path_first / max_path_driving_games
    min_speed = D(1) * max_path_first / max_path_driving_games
    max_speeds = {dn: max_speed for dn in duckie_names}
    min_speeds = {dn:  min_speed for dn in duckie_names}
    #max_waits = {dn: D(1) for dn in duckie_names}
    max_waits = {dn: D(1) for dn in duckie_names}

    # available_acc = [acc * max_path_first / max_path_driving_games for acc in [D(-2), D(-1), D(0), D(+1)]]
    available_acc = [acc * max_path_first / max_path_driving_games for acc in [D(-2), D(-1), D(0), D(+1)]]
    available_accels = {dn: frozenset(available_acc) for dn in duckie_names}
    light_actions = {dn: frozenset({NO_LIGHTS}) for dn in duckie_names}
    dt = D(1)
    initial_progress = {dn: 0 for dn in duckie_names}
    #collision_threshold = 3.0 * float(max_path_first / max_path_driving_games)
    collision_threshold = 3 * float(max_path_first / max_path_driving_games)
    #shared_resources_ds = D(1.5) * max_path_first / max_path_driving_games
    shared_resources_ds = D(1.5) * max_path_first / max_path_driving_games

# Parameters to compare solution with the game constructed in driving_games.zoo, get_sym()
two_player_duckie_game_parameters = DuckieGameParams(
    duckie_map=duckie_map,
    map_name=map_name,
    player_number=player_nb,
    player_names=duckie_names,
    duckie_geometries=duckie_geometries,
    max_speed=max_speeds,
    min_speed=min_speeds,
    max_wait=max_waits,
    max_path=max_paths,
    available_accels=available_accels,
    light_actions=light_actions,
    dt=dt,
    lanes=lanes,
    initial_progress=initial_progress,
    collision_threshold=collision_threshold,
    shared_resources_ds=shared_resources_ds
)


uncertainty_sets = UncertaintyParams(poss_monad=PossibilitySet(), mpref_builder=SetPreference1)
uncertainty_prob = UncertaintyParams(poss_monad=PossibilityDist(), mpref_builder=ProbPrefExpectedValue)


# Stretched version of the two player duckie game
with localcontext() as ctx:
    ctx.prec = 2
    map_name = 'maps/4way-stretched.yaml'
    duckie_map = load_duckie_map_from_yaml(os.path.join(module_path, map_name))
    player_nb = 2
    duckie_names = [PlayerName("Duckie_1"), PlayerName("Duckie_2")]
    lane_names = {
        duckie_names[0]: ['ls051', 'ls033', 'ls016'],
        #duckie_names[0]: ['ls026', 'ls022', 'L13'],
        #duckie_names[1]: ['ls041', 'ls036', 'ls026'],
        duckie_names[1]: ['ls041', 'ls035', 'ls050']
    }
    lanes: Dict[PlayerName, Lane]
    lanes = {dn: merge_lanes(get_lane_segments(duckie_map=duckie_map, lane_names=lane_names[dn])) for dn in duckie_names}

    duck_g = DuckieGeometry(
        mass=D(1000),
        length = D(4.5),
        width = D(1.8),
        color=(1, 0, 0),
        height=D(DB18().height)
    )

    duckie_geometries = {dn: duck_g for dn in duckie_names}

    max_paths = {dn: D(lanes[dn].get_lane_length()) * D(1) for dn in duckie_names}

    max_speed = D(5)
    min_speed = D(1)
    max_speeds = {dn: max_speed for dn in duckie_names}
    min_speeds = {dn:  min_speed for dn in duckie_names}
    max_waits = {dn: D(1) for dn in duckie_names}

    available_accels = {dn: frozenset([D(-2), D(-1), D(0), D(+1)]) for dn in duckie_names}
    light_actions = {dn: frozenset({NO_LIGHTS}) for dn in duckie_names}
    dt = D(1)
    initial_progress = {dn: 0 for dn in duckie_names}
    collision_threshold = 3
    shared_resources_ds = D(1.5)

# Parameters to compare solution with the game constructed in driving_games.zoo, get_sym()
two_player_duckie_game_parameters_stretched = DuckieGameParams(
    duckie_map=duckie_map,
    map_name=map_name,
    player_number=player_nb,
    player_names=duckie_names,
    duckie_geometries=duckie_geometries,
    max_speed=max_speeds,
    min_speed=min_speeds,
    max_wait=max_waits,
    max_path=max_paths,
    available_accels=available_accels,
    light_actions=light_actions,
    dt=dt,
    lanes=lanes,
    initial_progress=initial_progress,
    collision_threshold=collision_threshold,
    shared_resources_ds=shared_resources_ds
)



# Stretched version of a three player duckie game
with localcontext() as ctx:
    ctx.prec = 2
    map_name = 'maps/4way-stretched.yaml'
    duckie_map = load_duckie_map_from_yaml(os.path.join(module_path, map_name))
    player_nb = 3
    duckie_names = [PlayerName("Duckie_1"), PlayerName("Duckie_2"), PlayerName("Duckie_3")]
    lane_names = {
        duckie_names[0]: ['ls051', 'ls033', 'ls016'],
        duckie_names[1]: ['ls041', 'ls036', 'ls026'],
        duckie_names[2]: ['ls017', 'ls038', 'ls040']
    }
    lanes: Dict[PlayerName, Lane]
    lanes = {dn: merge_lanes(get_lane_segments(duckie_map=duckie_map, lane_names=lane_names[dn])) for dn in duckie_names}

    duck_g = DuckieGeometry(
        mass=D(1000),
        length = D(4.5),
        width = D(1.8),
        color=(1, 0, 0),
        height=D(DB18().height)
    )

    duckie_geometries = {dn: duck_g for dn in duckie_names}

    max_paths = {dn: D(lanes[dn].get_lane_length()) * D(1) for dn in duckie_names}

    max_speed = D(6)
    min_speed = D(0)
    max_speeds = {dn: max_speed for dn in duckie_names}
    min_speeds = {dn:  min_speed for dn in duckie_names}
    max_waits = {dn: D(5) for dn in duckie_names}

    #available_accels = {dn: frozenset([D(-1), D(0), D(+1)]) for dn in duckie_names}
    available_accels = {dn: frozenset([D(0), D(+3)]) for dn in duckie_names}
    #available_accels = {dn: frozenset([D(+1)]) for dn in duckie_names}
    #available_accels[duckie_names[1]] = frozenset([D(+1)])
    light_actions = {dn: frozenset({NO_LIGHTS}) for dn in duckie_names}
    dt = D(1)
    initial_progress = {dn: 0 for dn in duckie_names}
    collision_threshold = 3
    shared_resources_ds = D(1.5)

# A three player duckie game
three_player_duckie_game_parameters_stretched = DuckieGameParams(
    duckie_map=duckie_map,
    map_name=map_name,
    player_number=player_nb,
    player_names=duckie_names,
    duckie_geometries=duckie_geometries,
    max_speed=max_speeds,
    min_speed=min_speeds,
    max_wait=max_waits,
    max_path=max_paths,
    available_accels=available_accels,
    light_actions=light_actions,
    dt=dt,
    lanes=lanes,
    initial_progress=initial_progress,
    collision_threshold=collision_threshold,
    shared_resources_ds=shared_resources_ds
)