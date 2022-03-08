from dataclasses import dataclass
from typing import List, Mapping, Optional, Tuple

from shapely.geometry import Polygon

from dg_commons import PlayerName
from dg_commons.maps import DgLanelet
from dg_commons.planning import RefLaneGoal, RefLaneGoal
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.scenarios import DgScenario

__all__ = ["TrajectoryWorld"]


@dataclass
class TrajectoryWorld:
    """Object holding all info about the world"""

    map_name: str
    """ Map name for the world """
    scenario: DgScenario
    """ DgScenario """
    geo: Mapping[PlayerName, VehicleGeometry]
    """ Geometry of each player """
    lanes: Mapping[PlayerName, List[Tuple[DgLanelet, Optional[Polygon]]]]
    """ Reference lanes for each player """
    goals: Mapping[PlayerName, RefLaneGoal]
    """Goals for each player"""

    def __post_init__(self):
        assert (
            self.lanes.keys() == self.geo.keys() == self.goals.keys()
        ), f"Keys do not match: lanes = {self.lanes.keys()}, geo = {self.geo.keys()}"

    def get_players(self) -> List[PlayerName]:
        return list(self.geo.keys())

    def get_lanes(self, player: PlayerName) -> List[Tuple[DgLanelet, Optional[Polygon]]]:
        return self.lanes[player]

    def get_geometry(self, player: PlayerName) -> VehicleGeometry:
        return self.geo[player]
