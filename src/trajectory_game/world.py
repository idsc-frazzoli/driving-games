from dataclasses import dataclass
from typing import Mapping, Set

from games import PlayerName
from .paths import PathWithBounds

__all__ = ["World"]


@dataclass
class World:
    """ Object holding all info about the world """

    """ Will add more functionality when needed """
    # fixme just a set of possible path and maybe we assign later a path to each player
    ref: Mapping[PlayerName, PathWithBounds]
    """ Reference paths for each player """

    metrics: Set["Metric"]
    # [az] todo metrics as a property of the world?! mmmh...
    """ Metrics to evaluate for each outcome. """

    def get_reference(self, player: PlayerName) -> PathWithBounds:
        return self.ref[player]
