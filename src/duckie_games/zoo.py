import os

from decimal import Decimal as D, localcontext
from typing import Dict

from duckietown_world.world_duckietown.duckiebot import DB18

from games import PlayerName, UncertaintyParams
from possibilities import PossibilitySet, PossibilityDist
from preferences import SetPreference1
from preferences.preferences_probability import ProbPrefExpectedValue
from driving_games.structures import NO_LIGHTS

from world.utils import get_lane_segments, merge_lanes, Lane
from world.map_loading import load_driving_game_map
from duckie_games.game_generation import DuckieGameParams
from duckie_games.structures import DuckieGeometry

__all__ = ['two_player_duckie_game_parameters', 'three_player_duckie_game_parameters', 'uncertainty_sets', 'uncertainty_prob']

module_path = os.path.dirname(__file__)


uncertainty_sets = UncertaintyParams(poss_monad=PossibilitySet(), mpref_builder=SetPreference1)
uncertainty_prob = UncertaintyParams(poss_monad=PossibilityDist(), mpref_builder=ProbPrefExpectedValue)


# Stretched version of the two player duckie game
with localcontext() as ctx:
    ctx.prec = 2
    map_name = '4way'
    # map_name = '4way-double'
    duckie_map = load_driving_game_map(map_name)
    player_nb = 2
    duckie_names = [PlayerName("Duckie_1"), PlayerName("Duckie_2")]
    lane_names = { # 4way lanes
        duckie_names[0]: ['ls051', 'ls033', 'ls016'],
        #duckie_names[0]: ['ls026', 'ls022', 'L13'],
        #duckie_names[1]: ['ls041', 'ls036', 'ls026'],  # gives error
        duckie_names[1]: ['ls041', 'ls035', 'ls050']
    }
    # lane_names = { # 4way double
    #     duckie_names[0]: ['ls131', 'ls110', 'ls157'],
    #     duckie_names[1]: ['ls159', 'ls102', 'ls045']
    # }
    lanes: Dict[PlayerName, Lane]
    lanes = {dn: merge_lanes(get_lane_segments(duckie_map=duckie_map, lane_names=lane_names[dn])) for dn in duckie_names}

    duck_g = DuckieGeometry(  # 4way
        mass=D(1000),
        length = D(4.5),
        width = D(1.8),
        color=(1, 0, 0),
        height=D(DB18().height)
    )

    # duck_g = DuckieGeometry(  # 4way-double
    #     mass=D(1000),
    #     length=D(2.25),
    #     width=D(0.9),
    #     color=(1, 0, 0),
    #     height=D(DB18().height)
    # )

    duckie_geometries = {dn: duck_g for dn in duckie_names}

    max_paths = {dn: D(lanes[dn].get_lane_length()) * D(1) for dn in duckie_names}

    max_speed = D(5)
    min_speed = D(1)
    max_speeds = {dn: max_speed for dn in duckie_names}
    min_speeds = {dn:  min_speed for dn in duckie_names}
    max_waits = {dn: D(1) for dn in duckie_names}

    #available_accels = {dn: frozenset([D(-2), D(-1), D(0), D(+1)]) for dn in duckie_names}
    available_accels = {dn: frozenset([D(-1), D(0), D(+1)]) for dn in duckie_names}
    light_actions = {dn: frozenset({NO_LIGHTS}) for dn in duckie_names}
    dt = D(1)
    initial_progress = {dn: 0 for dn in duckie_names}
    collision_threshold = 3
    shared_resources_ds = duck_g.width / D(3)

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


# Stretched version of a three player duckie game
with localcontext() as ctx:
    ctx.prec = 2
    map_name = '4way'
    duckie_map = load_driving_game_map(map_name)
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

    # max_paths = {dn: D(lanes[dn].get_lane_length()) * D(1) for dn in duckie_names}
    max_paths = {dn: D(21) for dn in duckie_names}
    max_speed = D(3)
    min_speed = D(1)
    max_speeds = {dn: max_speed for dn in duckie_names}
    min_speeds = {dn:  min_speed for dn in duckie_names}
    max_waits = {dn: D(1) for dn in duckie_names}

    #available_accels = {dn: frozenset([D(-1), D(0), D(+1)]) for dn in duckie_names}
    available_accels = {dn: frozenset([D(0), D(+3)]) for dn in duckie_names}
    #available_accels = {dn: frozenset([D(+1)]) for dn in duckie_names}
    #available_accels[duckie_names[1]] = frozenset([D(+1)])
    light_actions = {dn: frozenset({NO_LIGHTS}) for dn in duckie_names}
    dt = D(1)
    initial_progress = {dn: 0 for dn in duckie_names}
    collision_threshold = 3
    shared_resources_ds = D(1)

# A three player duckie game
three_player_duckie_game_parameters = DuckieGameParams(
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