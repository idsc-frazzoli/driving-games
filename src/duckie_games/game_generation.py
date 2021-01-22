from dataclasses import dataclass
from decimal import Decimal as D
from typing import cast, Dict, FrozenSet, List
from frozendict import frozendict

from duckietown_world.world_duckietown.duckietown_map import DuckietownMap
from duckietown_world.world_duckietown.lane_segment import LaneSegment

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
from driving_games.structures import NO_LIGHTS, Lights, SE2_disc

from duckie_games.structures import DuckieGeometry, DuckieState, DuckieActions, DuckieCosts
from duckie_games.duckie_observations import DuckieObservation, DuckieDirectObservations
from duckie_games.rectangle import Rectangle
from duckie_games.duckie_dynamics import DuckieDynamics
from duckie_games.visualisation import DuckieGameVisualization
from duckie_games.personal_reward import DuckiePersonalRewardStructureTime
from duckie_games.joint_reward import DuckieJointReward
from duckie_games.preferences_coll_time import DuckiePreferencesCollTime
from duckie_games.utils import (
    interpolate_along_lane,
    from_SE2Transform_to_SE2_disc,
    LaneSegmentHashable,
    DuckietownMapHashable
)


DuckieGame = Game[DuckieState, DuckieActions, DuckieObservation, DuckieCosts, Collision, Rectangle]

DuckieGamePlayer = GamePlayer[
    DuckieState, DuckieActions, DuckieObservation, DuckieCosts, Collision, Rectangle
]

Lane = LaneSegment


@dataclass
class DuckieVehicleParams:
    max_speed: D
    min_speed: D
    max_wait: D
    available_accels: FrozenSet[D]
    collision_threshold: float
    light_actions: FrozenSet[Lights]
    dt: D


@dataclass
class DuckieGameParams:
    duckie_map: DuckietownMap
    """ Map where the players play """

    map_name: str
    """ The name of the map """

    player_number: int
    """ Number of Duckies competing with each other """

    player_names: List[PlayerName]
    """ List of all the player names """

    duckie_geometries: Dict[PlayerName, DuckieGeometry]
    """ Geometry parameters of duckies """

    max_speed: Dict[PlayerName, D]
    """ Maximal speed along lane """

    min_speed: Dict[PlayerName, D]
    """ Minimal speed along lane """

    max_wait: Dict[PlayerName, D]
    """ Maximal time at v=0 """

    max_path: Dict[PlayerName, D]
    """ Max path length of a player"""

    available_accels: Dict[PlayerName, FrozenSet[D]]
    """ Available accelerations available """

    light_actions: Dict[PlayerName, FrozenSet[Lights]]
    """ Available light action"""

    dt: D
    """ Discretization timestep """

    lanes: Dict[PlayerName, Lane]
    """ Which duckie follows which lane """

    initial_progress: Dict[PlayerName, float]
    """ Initial progress along the lane """

    collision_threshold: float
    """ Collision threshold """

    shared_resources_ds: D
    """ Shared resources"""  # todo

    def __post_init__(self):
        check_duckie_game_params(self)

    @property
    def refs(self) -> Dict[PlayerName, SE2_disc]:
        """ Reference frames of players (start of lane) """
        res = {}
        for player_name in self.player_names:
            lane = self.lanes[player_name]
            start = 0
            pose_SE2_transform = interpolate_along_lane(lane=lane, along_lane=start)
            res[player_name] = from_SE2Transform_to_SE2_disc(pose_SE2_transform)
        return res


def get_duckie_game(
        duckie_game_params: DuckieGameParams, uncertainty_params: UncertaintyParams
) -> DuckieGame:
    """
    Returns the game for a duckiebot
    """
    ps: PossibilityMonad = uncertainty_params.poss_monad

    duckie_map = duckie_game_params.duckie_map
    duckie_map_hashable = DuckietownMapHashable.initializor(duckie_map)

    duckie_players: Dict[PlayerName, DuckieGamePlayer] = {}
    shared_resources_ds = duckie_game_params.shared_resources_ds
    dt = duckie_game_params.dt

    refs = duckie_game_params.refs

    for duckie_name in duckie_game_params.player_names:

        max_path = duckie_game_params.max_path[duckie_name]
        max_speed = duckie_game_params.max_speed[duckie_name]
        min_speed = duckie_game_params.min_speed[duckie_name]
        max_wait = duckie_game_params.max_wait[duckie_name]
        ref = refs[duckie_name]
        available_accels = duckie_game_params.available_accels[duckie_name]
        light_actions = duckie_game_params.light_actions[duckie_name]
        duckie_geometry = duckie_game_params.duckie_geometries[duckie_name]
        lane = duckie_game_params.lanes[duckie_name]
        lane_hashable = LaneSegmentHashable.initializor(lane)

        duckie_x = DuckieState(
            duckie_map=duckie_map_hashable,
            ref=ref,
            x=D(duckie_game_params.initial_progress[duckie_name]),
            lane=lane_hashable,
            wait=D(0),
            v=duckie_game_params.min_speed[duckie_name],
            light=NO_LIGHTS
        )
        duckie_initial = ps.unit(duckie_x)

        duckie_dynamics = DuckieDynamics(
            max_speed=max_speed,
            max_wait=max_wait,
            available_accels=available_accels,
            max_path=max_path,
            ref=ref,
            lights_commands=light_actions,
            min_speed=min_speed,
            vg=duckie_geometry,
            shared_resources_ds=shared_resources_ds,
            poss_monad=ps,
        )

        duckie_personal_reward_structure = DuckiePersonalRewardStructureTime(max_path)

        duckie_ac = get_accessible_states(duckie_initial, duckie_personal_reward_structure, duckie_dynamics, dt)
        duckie_possible_states = cast(FrozenSet[DuckieState], frozenset(duckie_ac.nodes))

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
        collision_threshold=duckie_game_params.collision_threshold,
        geometries=duckie_game_params.duckie_geometries
    )

    game_visualization: DuckieGameVisualization[
        DuckieState, DuckieActions, DuckieObservation, DuckieCosts, Collision
    ]

    game_visualization = DuckieGameVisualization(
        duckie_map=duckie_map,
        map_name=duckie_game_params.map_name,
        geometries=duckie_game_params.duckie_geometries,
        ds=duckie_game_params.shared_resources_ds
    )
    game: DuckieGame

    game = Game(
        players=frozendict(duckie_players),
        ps=ps,
        joint_reward=joint_reward,
        game_visualization=game_visualization,
    )
    return game


def check_duckie_game_params(dg_params: DuckieGameParams) -> None:
    """ Checks if all game parameters are filled out"""
    lengths = map(len,
                  [dg_params.player_names,
                   dg_params.duckie_geometries,
                   dg_params.max_speed,
                   dg_params.min_speed,
                   dg_params.max_wait,
                   dg_params.max_path,
                   dg_params.available_accels,
                   dg_params.light_actions,
                   dg_params.lanes,
                   dg_params.initial_progress]
                  )
    check_player_lengths = lambda x: x == dg_params.player_number
    msg = f"Specify duckie game parameters for each player"
    assert all(map(check_player_lengths, lengths)), msg
