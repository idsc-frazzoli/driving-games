from typing import Union, Tuple, Mapping, FrozenSet
from dataclasses import dataclass
from decimal import Decimal as D

from duckietown_world.geo.transforms import SE2Transform
from driving_games.structures import VehicleGeometry, VehicleState, VehicleActions, VehicleCosts
from driving_games.vehicle_observation import VehicleDirectObservations
from driving_games.personal_reward import VehiclePersonalRewardStructureTime
from driving_games.preferences_coll_time import VehiclePreferencesCollTime
from driving_games.joint_reward import VehicleJointReward
from driving_games.collisions import Collision

from games.solve.solution_structures import SolverParams, GameFactorization, GamePlayerPreprocessed
from games_zoo.solvers import SolverSpec
from games.game_def import (
    Game,
    PlayerName,
    Combined,
    RJ,
    RP,
    SR,
    U,
    X,
    Y
)


@dataclass(frozen=True)
class DuckieCost(VehicleCosts):
    pass


@dataclass(frozen=True)
class DuckieGeometry(VehicleGeometry):
    color: Union[str, Tuple[float, float, float]]
    """ Color of Duckiebot, e.g. "red", "green", "blue" """
    height: D
    """ Duckie Height [m] """


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class DuckieState(VehicleState):
    # world_ref: SE2Transform
    # """ Absolute pose in Duckietown world """
    #
    # tile_ref: SE2Transform
    # """ Relative pose from current tile """
    #
    # lane_ref: SE2Transform
    # """ Relative pose from current lane """

    @property
    def x_current_lane(self) -> D:
        """Longitudinal position relative to the current lane"""
        # todo convert total longitudinal position self.x to the longitudinal position of the current lane
        return self.x


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class DuckieActions(VehicleActions):
    pass


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class DuckieObservation:
    pass


class DuckiePersonalRewardStructureTime(VehiclePersonalRewardStructureTime):
    def __init__(self, max_path: D):
        VehiclePersonalRewardStructureTime.__init__(self, max_path=max_path)


class DuckieDirectObservations(VehicleDirectObservations):
    def __init__(
        self,
        my_possible_states: FrozenSet[DuckieState],
        possible_states: Mapping[PlayerName, FrozenSet[DuckieState]]
    ):
        VehicleDirectObservations.__init__(
            self,
            my_possible_states=my_possible_states,
            possible_states=possible_states
        )


class DuckiePreferencesCollTime(VehiclePreferencesCollTime):
    def __init__(self):
        VehiclePreferencesCollTime.__init__(self)


class DuckieJointReward(VehicleJointReward):
    def __init__(
            self,
            collision_threshold: float,
            geometries: Mapping[PlayerName, DuckieGeometry]
    ):
        VehicleJointReward.__init__(
            self,
            collision_threshold=collision_threshold,
            geometries=geometries
        )
