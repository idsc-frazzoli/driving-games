from dataclasses import dataclass
from decimal import Decimal as D
from typing import Tuple, Mapping, Sequence, Union, Optional

from commonroad.scenario.scenario import Scenario
from shapely.geometry import Polygon

from dg_commons import PlayerName
from dg_commons.maps import DgLanelet
from dg_commons.sim import CollisionReportPlayer
from .vehicle_dynamics import VehicleTrackDynamicsParams
from .structures import VehicleTrackState, VehicleActions, VehicleTimeCost
from .vehicle_observation import VehicleObs
from games import GamePlayer, Game


class DrivingGame(
    Game[VehicleTrackState, VehicleActions, VehicleObs, VehicleTimeCost, CollisionReportPlayer, Polygon]):
    pass


class DrivingGamePlayer(
    GamePlayer[VehicleTrackState, VehicleActions, VehicleObs, VehicleTimeCost, CollisionReportPlayer, Polygon]
):
    pass


@dataclass
class DgSimpleParams:
    scenario: Scenario
    """A commonroad scenario"""
    ref_lanes: Mapping[PlayerName, DgLanelet]
    """Reference lanes"""
    progress: Mapping[PlayerName, Tuple[D, D]]
    """Initial and End progress along the reference Lanelet"""
    track_dynamics_param: VehicleTrackDynamicsParams
    """Dynamics the players"""
    col_check_dt: D
    """Discretization step for collision checking. A smart choice related to the solver's one is advisable"""
    min_safety_distance: float
    """Minimum safety distance for the joint cost of the players"""
    shared_resources_ds: D
    """Shared resources discretization resolution"""
    plot_limits: Optional[Union[str, Sequence[Sequence[float]]]] = None

    def __post__init__(self):
        assert self.ref_lanes.keys() == self.progress.keys()
        for progress in self.progress.values():
            assert progress[0] <= progress[1]
