from dataclasses import dataclass
from typing import Mapping, List, Set

from commonroad.scenario.scenario import Scenario
from dg_commons.planning.lanes import DgLanelet

from games import PlayerName
from .structures import VehicleGeometry

__all__ = ["TrajectoryWorld"]


@dataclass
class TrajectoryWorld:
    """ Object holding all info about the world """

    map_name: str
    """ Map name for the world """

    scenario: Scenario
    """ CommonRoad scenario """

    geo: Mapping[PlayerName, VehicleGeometry]
    """ Geometry of each player """

    lanes: Mapping[PlayerName, Set[DgLanelet]]
    """ Reference lanes for each player """

    def __post_init__(self):
        assert self.lanes.keys() == self.geo.keys(), (
            f"Keys do not match: lanes = {self.lanes.keys()}, geo = {self.geo.keys()}"
        )

    def get_players(self) -> List[PlayerName]:
        return list(self.geo.keys())

    def get_lanes(self, player: PlayerName) -> Set[DgLanelet]:
        return self.lanes[player]

    def get_geometry(self, player: PlayerName) -> VehicleGeometry:
        return self.geo[player]
