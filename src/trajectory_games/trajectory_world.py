from dataclasses import dataclass
from typing import Mapping, List, Dict

from games import PlayerName
from world import Lane
from .structures import VehicleGeometry

__all__ = ["TrajectoryWorld"]


@dataclass
class TrajectoryWorld:
    """ Object holding all info about the world """

    map_name: str
    """ Map name for the world """

    geo: Mapping[PlayerName, VehicleGeometry]
    """ Geometry of each player """

    lanes: Mapping[PlayerName, Lane]
    """ Reference lanes for each player """

    weights: Mapping[PlayerName, str]
    """ Weights config name for each player"""

    def __post_init__(self):
        assert self.lanes.keys() == self.geo.keys() == self.weights.keys(), \
            f"Keys do not match: lanes = {self.lanes.keys()}," \
            f" geo = {self.geo.keys()}, weights = {self.weights.keys()}"

    def get_players(self) -> List[PlayerName]:
        return list(self.geo.keys())

    def get_lane(self, player: PlayerName) -> Lane:
        return self.lanes[player]

    def get_geometry(self, player: PlayerName) -> VehicleGeometry:
        return self.geo[player]

    def get_weights(self, player: PlayerName) -> str:
        return self.weights[player]
