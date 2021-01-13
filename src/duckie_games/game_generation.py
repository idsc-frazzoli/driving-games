from dataclasses import dataclass
from decimal import Decimal as D
from typing import cast, Dict, FrozenSet as ASet, List, Optional, Union
from frozendict import frozendict

from duckietown_world.world_duckietown.duckiebot import DB18
from duckietown_world.geo.transforms import SE2Transform

from games import (
    Game,
    GamePlayer,
    get_accessible_states,
    JointRewardStructure,
    PlayerName,
    UncertaintyParams,
)
from possibilities import PossibilityMonad
from driving_games.collisions import Collision
from driving_games.structures import (
    NO_LIGHTS,
    VehicleCosts,
    SE2_disc,
)

from driving_games.game_generation import DrivingGame, DrivingGamePlayer, TwoVehicleSimpleParams

from .structures import (
    DuckieGeometry,
    DuckieObservation,
    DuckieState,
    DuckieActions,
    DuckiePersonalRewardStructureTime,
    DuckieDirectObservations,
    DuckiePreferencesCollTime,
    DuckieJointReward,
)
from .duckie_dynamics import DuckieDynamics
from .visualisation import DuckieGameVisualization


DuckieGame = DrivingGame  # todo create more specific class
DuckieGamePlayers = DrivingGamePlayer  # todo create more specific class


@dataclass
class DuckieVehicleParams(TwoVehicleSimpleParams):
    pass


@dataclass
class DuckieGameParams:
    player_number: int
    """ Number of Duckies competing with each other """

    player_names: Optional[List[PlayerName]] = None
    """ Optional list of all the players """

    initial_poses: Optional[Dict[PlayerName, Union[SE2_disc, SE2Transform]]] = None  # todo change to duckietown poses
    """ Initial state (pose) in Duckietown World"""

    def __post_init__(self):
        if self.player_names is not None:
            len_player_names = len(self.player_names)
            len_initial_poses = len(self.initial_poses)
            msg = (
                f'Player number ({self.player_number}) must match number of occurrences'
                f' of player names ({len_player_names}) and and initial poses ({len_initial_poses})'
            )
            assert self.player_number == len_player_names == len_initial_poses, msg
        else:
            """ Generate a two player game for """
            self.player_names = [PlayerName("N↑"), PlayerName("W←")]
            self.initial_poses = {
                self.player_names[0]: (D(11), D(0), D(+90)),  # origin at bottom left
                self.player_names[1]: (D(22), D(11), D(-180))
            }


def get_duckie_game(
        vehicles_params: DuckieVehicleParams, game_params: DuckieGameParams, uncertainty_params: UncertaintyParams
) -> DuckieGame:
    """
    Returns the game for a duckiebot
    """
    # todo generate duckiebot game
    ps: PossibilityMonad = uncertainty_params.poss_monad
    L = vehicles_params.side + vehicles_params.road + vehicles_params.side
    start = vehicles_params.side + vehicles_params.road_lane_offset
    max_path = L - 1
    max_speed = vehicles_params.max_speed
    min_speed = vehicles_params.min_speed
    max_wait = vehicles_params.max_wait
    dt = vehicles_params.dt
    available_accels = vehicles_params.available_accels

    duckie_names: List[PlayerName] = []
    geometries: Dict[PlayerName, DuckieGeometry] = {}
    duckie_players: Dict[PlayerName, DrivingGamePlayer] = {}

    for i in range(0, game_params.player_number):
        duckie_name = game_params.player_names[i]
        duckie_names.append(duckie_name)

        # todo define duckiebot geometry parameters
        # width = D(DB18().width)
        # length = D(DB18().length)
        height = D(DB18().height)
        mass = D(1000)
        length = D(4.5)
        width = D(1.8)    # p1_ref = SE2_from_xytheta([start, 0, np.pi / 2])
        duckie_ref = game_params.initial_poses[duckie_name]

        duckie_g = DuckieGeometry(mass=mass, width=width, length=length, color=(1, 0, 0), height=height)

        geometries[duckie_name] = duckie_g

        duckie_x = DuckieState(
            ref=duckie_ref, x=D(vehicles_params.first_progress), wait=D(0), v=min_speed, light=NO_LIGHTS
        )
        duckie_initial = ps.unit(duckie_x)

        duckie_dynamics = DuckieDynamics(
            max_speed=max_speed,
            max_wait=max_wait,
            available_accels=available_accels,
            max_path=max_path,
            ref=duckie_ref,
            lights_commands=vehicles_params.light_actions,
            min_speed=min_speed,
            vg=duckie_g,
            shared_resources_ds=vehicles_params.shared_resources_ds,
            poss_monad=ps,
        )

        duckie_personal_reward_structure = DuckiePersonalRewardStructureTime(max_path)

        duckie_ac = get_accessible_states(duckie_initial, duckie_personal_reward_structure, duckie_dynamics, dt)
        duckie_possible_states = cast(ASet[DuckieState], frozenset(duckie_ac.nodes))

        # logger.info("npossiblestates", p1=len(p1_possible_states), p2=len(p2_possible_states))
        duckie_observations = DuckieDirectObservations(duckie_possible_states, {duckie_name: duckie_possible_states})

        duckie_preferences = DuckiePreferencesCollTime()
        duckie_player = GamePlayer(
            initial=duckie_initial,
            dynamics=duckie_dynamics,
            observations=duckie_observations,
            personal_reward_structure=duckie_personal_reward_structure,
            preferences=duckie_preferences,
            monadic_preference_builder=uncertainty_params.mpref_builder,
        )

        duckie_players[duckie_name] = duckie_player

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
        players=frozendict(duckie_players),
        ps=ps,
        joint_reward=joint_reward,
        game_visualization=game_visualization,
    )
    return game
