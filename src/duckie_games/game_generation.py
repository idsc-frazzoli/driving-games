from dataclasses import dataclass
from decimal import Decimal as D
from typing import cast, Dict, FrozenSet, FrozenSet as ASet, List
from frozendict import frozendict

from duckietown_world.world_duckietown.duckiebot import DB18

from games import (
    Game,
    GamePlayer,
    GameVisualization,
    get_accessible_states,
    JointRewardStructure,
    PlayerName,
    UncertaintyParams,
)
from possibilities import PossibilityMonad
from driving_games.collisions import Collision
from driving_games.joint_reward import VehicleJointReward
from driving_games.personal_reward import VehiclePersonalRewardStructureTime
from driving_games.preferences_coll_time import VehiclePreferencesCollTime
from driving_games.rectangle import Rectangle
from driving_games.structures import (
    Lights,
    NO_LIGHTS,
    VehicleActions,
    VehicleCosts,
    VehicleGeometry,
    VehicleState,
)
from driving_games.vehicle_dynamics import VehicleDynamics
from driving_games.vehicle_observation import VehicleDirectObservations, VehicleObservation
from driving_games.visualization import DrivingGameVisualization
from driving_games.game_generation import DrivingGame, DrivingGamePlayer, TwoVehicleSimpleParams

from .structures import (
    DuckieGeometry,
    DuckieObservation,
    DuckieCost,
    DuckieState,
    DuckieActions,
    DuckiePersonalRewardStructureTime,
    DuckieDirectObservations,
    DuckiePreferencesCollTime,
    DuckieJointReward,
)
from .duckie_dynamics import DuckieDynamics
from .visualisation import DuckieGameVisualization


DuckieGame = DrivingGame # todo create more specific class
DuckieGamePlayers = DrivingGamePlayer #todo create more specific class


@dataclass
class DuckieVehicleParams(TwoVehicleSimpleParams):
    #todo create class with only the relevant parameters
    player_number: int
    """ Number of Duckies competing with each other """


def get_duckie_game(
        vehicles_params: DuckieVehicleParams, uncertainty_params: UncertaintyParams
) -> DuckieGame:
    """
    Returns the game for a duckiebot
    """
    # todo generate duckiebot game
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

    # players: List[PlayerName] =[]
    #
    # for _ in range(0, vehicles_params.player_number):
    #     players.append(PlayerName(f"Duckie_{_}"))

    P2 = PlayerName("W←")
    P1 = PlayerName("N↑")

    # todo define duckiebot geometry parameters
    # width = D(DB18().width)
    # length = D(DB18().length)
    height = D(DB18().height)
    mass = D(1000)
    length = D(4.5)
    width = D(1.8)

    g1 = DuckieGeometry(mass=mass, width=width, length=length, color=(1, 0, 0), height=height)
    g2 = DuckieGeometry(mass=mass, width=width, length=length, color=(0, 0, 1), height=height)
    geometries = {P1: g1, P2: g2}
    p1_x = DuckieState(
        ref=p1_ref, x=D(vehicles_params.first_progress), wait=D(0), v=min_speed, light=NO_LIGHTS
    )
    p1_initial = ps.unit(p1_x)
    p2_x = DuckieState(
        ref=p2_ref, x=D(vehicles_params.second_progress), wait=D(0), v=min_speed, light=NO_LIGHTS
    )
    p2_initial = ps.unit(p2_x)
    p1_dynamics = DuckieDynamics(
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
    p2_dynamics = DuckieDynamics(
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
    p1_personal_reward_structure = DuckiePersonalRewardStructureTime(max_path)
    p2_personal_reward_structure = DuckiePersonalRewardStructureTime(max_path)

    g1 = get_accessible_states(p1_initial, p1_personal_reward_structure, p1_dynamics, dt)
    p1_possible_states = cast(ASet[DuckieState], frozenset(g1.nodes))
    g2 = get_accessible_states(p2_initial, p2_personal_reward_structure, p2_dynamics, dt)
    p2_possible_states = cast(ASet[DuckieState], frozenset(g2.nodes))

    # logger.info("npossiblestates", p1=len(p1_possible_states), p2=len(p2_possible_states))
    p1_observations = DuckieDirectObservations(p1_possible_states, {P2: p2_possible_states})
    p2_observations = DuckieDirectObservations(p2_possible_states, {P1: p1_possible_states})

    p1_preferences = DuckiePreferencesCollTime()
    p2_preferences = DuckiePreferencesCollTime()
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
    joint_reward: JointRewardStructure[DuckieState, DuckieActions, Collision]

    joint_reward = DuckieJointReward(
        collision_threshold=vehicles_params.collision_threshold, geometries=geometries
    )

    game_visualization: DuckieGameVisualization[
        DuckieState, DuckieActions, DuckieObservation, VehicleCosts, Collision
    ]
    game_visualization = DuckieGameVisualization(
        vehicles_params, L, geometries=geometries, ds=vehicles_params.shared_resources_ds
    )
    game: DuckieGame

    game = Game(
        players=frozendict(players),
        ps=ps,
        joint_reward=joint_reward,
        game_visualization=game_visualization,
    )
    return game