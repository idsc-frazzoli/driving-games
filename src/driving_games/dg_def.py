from dataclasses import dataclass
from decimal import Decimal as D
from typing import Tuple, Mapping, Sequence, Union, Optional

from commonroad.scenario.scenario import Scenario
from shapely.geometry import Polygon

from dg_commons import PlayerName
from dg_commons.maps import DgLanelet
from dg_commons.sim import CollisionReportPlayer
from driving_games import VehicleTrackDynamicsParams
from driving_games.structures import VehicleTrackState, VehicleActions, VehicleCosts
from driving_games.vehicle_observation import VehicleObs
from games import GamePlayer, Game


class DrivingGame(Game[VehicleTrackState, VehicleActions, VehicleObs, VehicleCosts, CollisionReportPlayer, Polygon]):
    pass


class DrivingGamePlayer(
    GamePlayer[VehicleTrackState, VehicleActions, VehicleObs, VehicleCosts, CollisionReportPlayer, Polygon]
):
    pass


@dataclass
class DGSimpleParams:
    game_dt: D
    """Game discretization"""
    scenario: Scenario  # fixme maybe a string to be loaded
    """A commonroad scenario"""
    ref_lanes: Mapping[PlayerName, DgLanelet]
    """Reference lanes"""
    progress: Mapping[PlayerName, Tuple[D, D]]
    """Initial and End progress along the reference Lanelet"""
    track_dynamics_param: VehicleTrackDynamicsParams
    """Dynamics the players"""
    shared_resources_ds: D
    plot_limits: Optional[Union[str, Sequence[Sequence[float]]]] = None

    def __post__init__(self):
        assert self.ref_lanes.keys() == self.progress.keys()
        for progress in self.progress.values():
            assert progress[0] <= progress[1]
