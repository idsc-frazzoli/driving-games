from decimal import Decimal as D

import duckietown_world as dw
from duckietown_world.world_duckietown.duckiebot import DB18
from games import PlayerName, UncertaintyParams
from possibilities import PossibilitySet, PossibilityDist
from preferences import SetPreference1
from preferences.preferences_probability import ProbPrefExpectedValue

from driving_games.structures import NO_LIGHTS

from duckie_games.utils import get_lane_segments, merge_lanes
from duckie_games.game_generation import DuckieGameParams
from duckie_games.structures import DuckieGeometry

__all__ = ['two_player_reference_game_parameters', 'uncertainty_sets', 'uncertainty_prob']


duckie_map = dw.load_map('4way')
player_nb = 2
duckie_names = [PlayerName("N↑"), PlayerName("W←")]
lane_names = {
    duckie_names[0]: ['ls051', 'ls033', 'ls016'],
    duckie_names[1]: ['ls041', 'ls036', 'ls026']
}
lanes = {dn: merge_lanes(get_lane_segments(duckie_map=duckie_map, lane_names=lane_names[dn])) for dn in duckie_names}

# duck_g = DuckieGeometry(
#         mass=D(1000),  # todo
#         width=D(DB18().width),
#         length=D(DB18().length),
#         color=(1, 0, 0),
#         height=D(DB18().height)
# )

duck_g = DuckieGeometry(
        mass=D(1000),  # todo
        width=D(1.8),
        length=D(4.5),
        color=(1, 0, 0),
        height=D(DB18().height)
)

duckie_geometries = {dn: duck_g for dn in duckie_names}

max_speeds = {dn: D(5) for dn in duckie_names}
min_speeds = {dn:  D(1) for dn in duckie_names}
max_waits = {dn: D(1) for dn in duckie_names}
max_paths = {dn: D(21) for dn in duckie_names}

available_accels = {dn: frozenset({D(-2), D(-1), D(0), D(+1)}) for dn in duckie_names}
light_actions = {dn: frozenset({NO_LIGHTS}) for dn in duckie_names}
dt = D(1)
initial_progress = {dn: 0 for dn in duckie_names}
collision_threshold = 3.0
shared_resources_ds = D(1.5)


# Parameters to compare solution with the original game
two_player_reference_game_parameters = DuckieGameParams(
    duckie_map=duckie_map,
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
