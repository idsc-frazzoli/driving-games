from dataclasses import dataclass
from typing import Mapping

from games import PlayerName
from .paths import PathWithBounds

__all__ = ["World"]


@dataclass
class World:
    """ Object holding all info about the world """
    """ Will add more functionality when needed """

    ref: Mapping[PlayerName, PathWithBounds]

    def get_reference(self, player: PlayerName) -> PathWithBounds:
        return self.ref[player]
