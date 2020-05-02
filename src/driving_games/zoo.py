from decimal import Decimal as D
from typing import Dict

from games import Game, GameSpec
from possibilities import One
from .collisions import Collision
from .game_generation import get_two_vehicle_game, TwoVehicleSimpleParams
from .structures import (
    NO_LIGHTS,
    VehicleActions,
    VehicleCosts,
    VehicleState,
)
from .vehicle_observation import VehicleObservation


def get_game1() -> Game[One, VehicleState, VehicleActions, VehicleObservation, VehicleCosts, Collision]:
    p = TwoVehicleSimpleParams(
        side=D(8),
        road=D(6),
        road_lane_offset=D(4),
        max_speed=D(5),
        min_speed=D(1),
        max_wait=D(1),
        # available_accels={D(-2), D(0), D(+1)},
        available_accels=frozenset({D(-2), D(-1), D(0), D(+1)}),
        collision_threshold=3.0,
        light_actions=frozenset({NO_LIGHTS}),
        dt=D(1),
        first_progress=D(0),
        second_progress=D(0),
    )
    return get_two_vehicle_game(p)


def get_game2() -> Game[One, VehicleState, VehicleActions, VehicleObservation, VehicleCosts, Collision]:
    p = TwoVehicleSimpleParams(
        side=D(8),
        road=D(6),
        road_lane_offset=D(3),  # center
        max_speed=D(5),
        min_speed=D(1),
        max_wait=D(1),
        # available_accels={D(-2), D(0), D(+1)},
        available_accels=frozenset({D(-2), D(-1), D(0), D(+1)}),
        collision_threshold=3.0,
        light_actions=frozenset({NO_LIGHTS}),
        dt=D(1),
        first_progress=D(0),
        second_progress=D(0),
    )
    return get_two_vehicle_game(p)


driving_games_zoo: Dict[str, GameSpec] = {}

driving_games_zoo["asym"] = GameSpec(
    desc="""
Slightly asymmetric case. West is advantaged.  
""",
    game=get_game1(),
)

driving_games_zoo["sym"] = GameSpec(
    desc="""
Super symmetric case

""",
    game=get_game2(),
)
