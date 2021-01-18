from typing import Union, Tuple, Mapping, FrozenSet, Sequence
from dataclasses import dataclass
from decimal import Decimal as D
import numpy as np

import geometry as geo
from duckietown_world.geo.transforms import SE2Transform
from duckietown_world.world_duckietown.lane_segment import LaneSegment
from duckietown_world.world_duckietown.duckietown_map import DuckietownMap

from driving_games.structures import VehicleGeometry, VehicleState, VehicleActions, VehicleCosts, SE2_disc, Lights
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

from duckie_games.utils import interpolate_along_lane, from_SE2Transform_to_SE2_disc


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
class DuckieState:
    duckie_map: DuckietownMap
    """ Duckietown world map where the duckie is playing """

    ref: SE2_disc
    """ Reference frame from where the vehicle started """

    lane: LaneSegment
    """ Lane that the duckie follows"""

    x: D
    """ Position along lane """

    v: D
    """ Velocity along lane """

    wait: D
    """ How long we have been at speed = 0. We want to keep track so bound this. """

    light: Lights
    """ The current lights signal. """

    __print_order__ = ["x", "v"]  # only print these attributes

    @property
    def abs_pose(self) -> SE2_disc:
        """ get the absolute pose of the duckie in the map """

        pose_SE2_transform = interpolate_along_lane(lane=self.lane, along_lane=float(self.x))
        return from_SE2Transform_to_SE2_disc(pose_SE2_transform)

    @property
    def ref_pose(self) -> SE2_disc:
        """ get the pose of the duckie relative to the reference frame"""

        # Get the SE2 representation of the absolute pose
        *t_abs, theta_abs_deg = map(float, self.abs_pose)
        theta_abs_rad = np.deg2rad(theta_abs_deg)
        q_abs = geo.SE2_from_translation_angle(t_abs, theta_abs_rad)

        # Get SE2 representation of the ref pose
        *t_ref, theta_ref_deg = map(float, self.ref)
        theta_ref_rad = np.deg2rad(theta_ref_deg)
        q_ref = geo.SE2_from_translation_angle(t_ref, theta_ref_rad)

        # Get the the pose of the duckie in the reference frame
        q_abs_from_q_ref = geo.SE2.multiply(geo.SE2.inverse(q_ref), q_abs)
        t, theta_rad = geo.translation_angle_from_SE2(q_abs_from_q_ref)
        x, y = t
        theta_deg = np.rad2deg(theta_rad)
        return (D(x), D(y), D(theta_deg))


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
