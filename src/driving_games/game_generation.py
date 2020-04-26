from dataclasses import dataclass
from decimal import Decimal as D
from typing import FrozenSet as ASet, Mapping

from driving_games import logger
from driving_games.access import get_accessible_states
from driving_games.driving_example import (
    Lights,
    NO_LIGHTS,
    RJ_,
    RP_,
    U_,
    VehicleDirectObservations,
    VehicleDynamics,
    VehicleJointReward,
    VehiclePersonalRewardStructureTime,
    VehiclePreferencesCollTime,
    VehicleState,
    X_,
    Y_,
)
from games import Game, GamePlayer, JointRewardStructure, PlayerName
from preferences import SetPreference1


@dataclass
class TwoVehicleSimpleParams:
    side: D
    road: D
    road_lane_offset: D
    max_speed: D
    min_speed: D
    max_wait: D
    available_accels: ASet[D]
    collision_threshold: float
    light_actions: ASet[Lights]
    dt: D
    # initial positions
    first_progress: D
    second_progress: D


def get_game1() -> Game:
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
        first_progress=D(2),
        second_progress=D(0),
    )
    return get_two_vehicle_game(p)


def get_two_vehicle_game(params: TwoVehicleSimpleParams) -> Game:
    L = params.side + params.road + params.side
    start = params.side + params.road_lane_offset
    max_path = L - 1
    # p1_ref = SE2_from_xytheta([start, 0, np.pi / 2])
    p1_ref = (D(start), D(0), D(+90))
    # p2_ref = SE2_from_xytheta([L, start, -np.pi])
    p2_ref = (D(L - 1), D(start), D(-180))
    max_speed = params.max_speed
    min_speed = params.min_speed
    max_wait = params.max_wait
    dt = params.dt
    available_accels = params.available_accels

    P1 = PlayerName("p1")
    P2 = PlayerName("p2")
    p1_initial = frozenset(
        {VehicleState(ref=p1_ref, x=D(params.first_progress), wait=D(0), v=min_speed, light="none")}
    )
    p2_initial = frozenset(
        {
            VehicleState(
                ref=p2_ref, x=D(params.second_progress), wait=D(0), v=min_speed, light="none"
            )
        }
    )
    p1_dynamics = VehicleDynamics(
        max_speed=max_speed,
        max_wait=max_wait,
        available_accels=available_accels,
        max_path=max_path,
        ref=p1_ref,
        lights_commands=params.light_actions,
        min_speed=min_speed,
    )
    p2_dynamics = VehicleDynamics(
        min_speed=min_speed,
        max_speed=max_speed,
        max_wait=max_wait,
        available_accels=available_accels,
        max_path=max_path,
        ref=p2_ref,
        lights_commands=params.light_actions,
    )
    p1_personal_reward_structure = VehiclePersonalRewardStructureTime(max_path)
    p2_personal_reward_structure = VehiclePersonalRewardStructureTime(max_path)

    g1 = get_accessible_states(p1_initial, p1_personal_reward_structure, p1_dynamics, dt)
    p1_possible_states = frozenset(g1.nodes)
    g2 = get_accessible_states(p2_initial, p2_personal_reward_structure, p2_dynamics, dt)
    p2_possible_states = frozenset(g2.nodes)

    logger.info("npossiblestates", p1=len(p1_possible_states), p2=len(p2_possible_states))
    p1_observations = VehicleDirectObservations(p1_possible_states, {P2: p2_possible_states})
    p2_observations = VehicleDirectObservations(p2_possible_states, {P1: p1_possible_states})

    p1_preferences = VehiclePreferencesCollTime()
    p2_preferences = VehiclePreferencesCollTime()
    set_preference_aggregator = SetPreference1
    p1 = GamePlayer(
        initial=p1_initial,
        dynamics=p1_dynamics,
        observations=p1_observations,
        personal_reward_structure=p1_personal_reward_structure,
        preferences=p1_preferences,
        set_preference_aggregator=set_preference_aggregator,
    )
    p2 = GamePlayer(
        initial=p2_initial,
        dynamics=p2_dynamics,
        observations=p2_observations,
        personal_reward_structure=p2_personal_reward_structure,
        preferences=p2_preferences,
        set_preference_aggregator=set_preference_aggregator,
    )
    players: Mapping[PlayerName, GamePlayer[X_, U_, Y_, RP_, RJ_]]
    players = {P1: p1, P2: p2}
    joint_reward: JointRewardStructure[X_, U_, RJ_]
    joint_reward = VehicleJointReward(collision_threshold=params.collision_threshold)

    game: Game[X_, U_, Y_, RP_, RJ_] = Game(players, joint_reward)
    return game
