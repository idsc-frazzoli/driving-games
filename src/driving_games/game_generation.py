from dataclasses import dataclass
from decimal import Decimal as D
from typing import cast, Dict, FrozenSet, FrozenSet as ASet

from frozendict import frozendict

from games import (
    Game,
    GamePlayer,
    GameVisualization,
    get_accessible_states,
    JointRewardStructure,
    PlayerName,
)
from possibilities import One, PossibilityStructure, ProbabilitySet
from preferences import SetPreference1
from .driving_example import VehiclePersonalRewardStructureTime
from .joint_reward import VehicleJointReward
from .pref_coll_time import VehiclePreferencesCollTime
from .structures import (
    CollisionCost,
    Lights,
    VehicleActions,
    VehicleCosts,
    VehicleDirectObservations,
    VehicleDynamics,
    VehicleObservation,
    VehicleState,
)
from .visualization import DrivingGameVisualization

DrivingGame = Game[One, VehicleState, VehicleActions, VehicleObservation, VehicleCosts, CollisionCost]
DrivingGamePlayer = GamePlayer[
    One, VehicleState, VehicleActions, VehicleObservation, VehicleCosts, CollisionCost
]


@dataclass
class TwoVehicleSimpleParams:
    side: D
    road: D
    road_lane_offset: D
    max_speed: D
    min_speed: D
    max_wait: D
    available_accels: FrozenSet[D]
    collision_threshold: float
    light_actions: FrozenSet[Lights]
    dt: D
    # initial positions
    first_progress: D
    second_progress: D


def get_two_vehicle_game(params: TwoVehicleSimpleParams,) -> DrivingGame:
    ps: PossibilityStructure[One] = ProbabilitySet()
    L = params.side + params.road + params.side
    start = params.side + params.road_lane_offset
    max_path = L - 1
    # p1_ref = SE2_from_xytheta([start, 0, np.pi / 2])
    p1_ref = (D(start), D(0), D(+90))
    # p2_ref = SE2_from_xytheta([L, start, -np.pi])
    p2_ref = (D(L), D(start), D(-180))
    max_speed = params.max_speed
    min_speed = params.min_speed
    max_wait = params.max_wait
    dt = params.dt
    available_accels = params.available_accels

    # P1 = PlayerName("👩‍🦰")  # "👩🏿")
    # P2 = PlayerName("👳🏾‍")
    P1 = PlayerName("p1")
    P2 = PlayerName("p2")
    P2 = PlayerName("⬅")
    P1 = PlayerName("⬆")

    p1_initial = ps.lift_one(
        VehicleState(ref=p1_ref, x=D(params.first_progress), wait=D(0), v=min_speed, light="none")
    )
    p2_initial = ps.lift_one(
        VehicleState(ref=p2_ref, x=D(params.second_progress), wait=D(0), v=min_speed, light="none")
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
    p1_possible_states = cast(ASet[VehicleState], frozenset(g1.nodes))
    g2 = get_accessible_states(p2_initial, p2_personal_reward_structure, p2_dynamics, dt)
    p2_possible_states = cast(ASet[VehicleState], frozenset(g2.nodes))

    # logger.info("npossiblestates", p1=len(p1_possible_states), p2=len(p2_possible_states))
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
    players: Dict[PlayerName, DrivingGamePlayer]
    players = {P1: p1, P2: p2}
    joint_reward: JointRewardStructure[VehicleState, VehicleActions, CollisionCost]
    joint_reward = VehicleJointReward(collision_threshold=params.collision_threshold)

    game_visualization: GameVisualization[
        One, VehicleState, VehicleActions, VehicleObservation, VehicleCosts, CollisionCost
    ]
    game_visualization = DrivingGameVisualization(params, L)
    game: DrivingGame

    game = Game(
        players=frozendict(players), ps=ps, joint_reward=joint_reward, game_visualization=game_visualization
    )
    return game
