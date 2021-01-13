from decimal import Decimal as D

from games import GameSpec, UncertaintyParams
from possibilities import PossibilitySet, PossibilityDist
from preferences import SetPreference1
from preferences.preferences_probability import ProbPrefExpectedValue
from duckie_games.game_generation import get_duckie_game, DuckieVehicleParams
from driving_games.structures import NO_LIGHTS

road = D(6)
duckie_vehicle_parameters = DuckieVehicleParams(
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
    player_number=2
)
uncertainty_sets = UncertaintyParams(poss_monad=PossibilitySet(), mpref_builder=SetPreference1)
uncertainty_prob = UncertaintyParams(poss_monad=PossibilityDist(), mpref_builder=ProbPrefExpectedValue)

def test_game_generation():
    duckie_game_sets = get_duckie_game(duckie_vehicle_parameters, uncertainty_sets)
    duckie_game_prop = get_duckie_game(duckie_vehicle_parameters, uncertainty_prob)