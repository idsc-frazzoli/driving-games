from typing import Optional, Callable, Mapping
from dataclasses import dataclass
import decimal as D

from duckietown_world.geo.transforms import SE2Transform
from driving_games.structures import VehicleGeometry, VehicleState, VehicleActions, VehicleCosts
from games.solve.solution_structures import SolverParams, GameFactorization, GamePlayerPreprocessed
from games_zoo.solvers import SolverSpec
from games.game_def import (
    Game,
    PlayerName,
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
    color: str
    """ Color of Duckiebot, e.g. "red", "green", "blue" """
    height: D
    """ Duckie Hight [m] """


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class DuckieState(VehicleState):
    world_ref: SE2Transform
    """ Absolute pose in Duckietown world """

    tile_ref: SE2Transform
    """ Relative pose from current tile """

    lane_ref: SE2Transform
    """ Relative pose from current lane """

    @property
    def x_current_lane(self) -> D:
        """Longitudinal position relative to the current lane"""
        #todo convert total longitudinal position self.x to the longitudinal position of the current lane
        return self.x


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class DuckieActions(VehicleActions):
    pass