from dataclasses import dataclass
from typing import Mapping

from games import PlayerName
from .structures import VehicleGeometry
from .paths import PathWithBounds

__all__ = ["World"]


@dataclass
class World:
    """ Object holding all info about the world """

    # fixme just a set of possible path and maybe we assign later a path to each player
    ref: Mapping[PlayerName, PathWithBounds]
    """ Reference paths for each player """

    geo: Mapping[PlayerName, VehicleGeometry]
    """ Geometry of each player """

    def get_reference(self, player: PlayerName) -> PathWithBounds:
        return self.ref[player]

    def get_geometry(self, player: PlayerName) -> VehicleGeometry:
        return self.geo[player]
