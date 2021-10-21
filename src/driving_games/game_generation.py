from dataclasses import dataclass
from decimal import Decimal as D
from typing import cast, Dict, FrozenSet, FrozenSet as ASet

from frozendict import frozendict
from shapely.geometry import Polygon

from dg_commons import PlayerName
from dg_commons.sim import CollisionReportPlayer
from dg_commons.sim.models.vehicle_ligths import LightsCmd, NO_LIGHTS
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from games import (
    Game,
    GamePlayer,
    GameVisualization,
    get_accessible_states,
    JointRewardStructure,
    UncertaintyParams,
)
from possibilities import PossibilityMonad
from .joint_reward import VehicleJointReward
from .personal_reward import VehiclePersonalRewardStructureTime
from .preferences_coll_time import VehiclePreferencesCollTime
from .structures import (
    VehicleActions,
    VehicleCosts,
    VehicleState,
)
from .vehicle_dynamics import VehicleTrackDynamics
from .vehicle_observation import VehicleDirectObservations, VehicleObservation
from .visualization import DrivingGameVisualization

DrivingGame = Game[VehicleState, VehicleActions, VehicleObservation, VehicleCosts, CollisionReportPlayer, Polygon]
DrivingGamePlayer = GamePlayer[
    VehicleState, VehicleActions, VehicleObservation, VehicleCosts, CollisionReportPlayer, Polygon
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
    light_actions: FrozenSet[LightsCmd]
    dt: D
    # initial positions
    first_progress: D
    second_progress: D
    shared_resources_ds: D


def get_two_vehicle_game(vehicles_params: TwoVehicleSimpleParams, uncertainty_params: UncertaintyParams) -> DrivingGame:
    ps: PossibilityMonad = uncertainty_params.poss_monad
    L = vehicles_params.side + vehicles_params.road + vehicles_params.side
    start = vehicles_params.side + vehicles_params.road_lane_offset
    max_path = L - 1
    # p1_ref = SE2_from_xytheta([start, 0, np.pi / 2])
    p1_ref = (D(start), D(0), D(+90))
    # p2_ref = SE2_from_xytheta([L, start, -np.pi])
    p2_ref = (D(L), D(start), D(-180))
    max_speed = vehicles_params.max_speed
    min_speed = vehicles_params.min_speed
    max_wait = vehicles_params.max_wait
    dt = vehicles_params.dt
    available_accels = vehicles_params.available_accels

    # P1 = PlayerName("👩‍🦰")  # "👩🏿")
    # P2 = PlayerName("👳🏾‍")
    # P1 = PlayerName("p1")
    # P2 = PlayerName("p2")
    # P2 = PlayerName("⬅")
    # P1 = PlayerName("⬆")
    P2 = PlayerName("W←")
    P1 = PlayerName("N↑")
    mass = D(1000)
    length = D(4.5)
    width = D(1.8)

    g1 = VehicleGeometry.default_car(color=(1, 0, 0))
    g2 = VehicleGeometry.default_car(color=(0, 0, 1))
    geometries = {P1: g1, P2: g2}
    p1_x = VehicleState(ref=p1_ref, x=D(vehicles_params.first_progress), wait=D(0), v=min_speed, light=NO_LIGHTS)
    p1_initial = ps.unit(p1_x)
    p2_x = VehicleState(ref=p2_ref, x=D(vehicles_params.second_progress), wait=D(0), v=min_speed, light=NO_LIGHTS)
    p2_initial = ps.unit(p2_x)
    p1_dynamics = VehicleTrackDynamics(
        max_speed=max_speed,
        max_wait=max_wait,
        available_accels=available_accels,
        max_path=max_path,
        ref=p1_ref,
        lights_commands=vehicles_params.light_actions,
        min_speed=min_speed,
        vg=g1,
        shared_resources_ds=vehicles_params.shared_resources_ds,
        poss_monad=ps,
    )
    p2_dynamics = VehicleTrackDynamics(
        min_speed=min_speed,
        max_speed=max_speed,
        max_wait=max_wait,
        available_accels=available_accels,
        max_path=max_path,
        ref=p2_ref,
        lights_commands=vehicles_params.light_actions,
        vg=g2,
        shared_resources_ds=vehicles_params.shared_resources_ds,
        poss_monad=ps,
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
    p1 = GamePlayer(
        initial=p1_initial,
        dynamics=p1_dynamics,
        observations=p1_observations,
        personal_reward_structure=p1_personal_reward_structure,
        preferences=p1_preferences,
        monadic_preference_builder=uncertainty_params.mpref_builder,
    )
    p2 = GamePlayer(
        initial=p2_initial,
        dynamics=p2_dynamics,
        observations=p2_observations,
        personal_reward_structure=p2_personal_reward_structure,
        preferences=p2_preferences,
        monadic_preference_builder=uncertainty_params.mpref_builder,
    )
    players: Dict[PlayerName, DrivingGamePlayer]
    players = {P1: p1, P2: p2}
    joint_reward: JointRewardStructure[VehicleState, VehicleActions, Collision]

    joint_reward = VehicleJointReward(collision_threshold=vehicles_params.collision_threshold, geometries=geometries)

    game_visualization: GameVisualization[VehicleState, VehicleActions, VehicleObservation, VehicleCosts, Collision]
    game_visualization = DrivingGameVisualization(
        vehicles_params, L, geometries=geometries, ds=vehicles_params.shared_resources_ds
    )
    game: DrivingGame

    game = Game(
        players=frozendict(players),
        ps=ps,
        joint_reward=joint_reward,
        game_visualization=game_visualization,
    )
    return game
