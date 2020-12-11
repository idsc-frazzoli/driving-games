from dataclasses import dataclass
from typing import Mapping, Set

from games import PlayerName
from .paths import PathWithBounds

__all__ = ["World"]


@dataclass
class World:
    """ Object holding all info about the world """

    """ Will add more functionality when needed """

    ref: Mapping[PlayerName, PathWithBounds]
    """ Reference paths for each player """

    metrics: Set["Metric"]
    """ Metrics to evaluate for each outcome. """

    def get_reference(self, player: PlayerName) -> PathWithBounds:
        return self.ref[player]
