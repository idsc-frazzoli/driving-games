from decimal import Decimal as D
from typing import Mapping

from games import PlayerName

from driving_games.visualization import DrivingGameVisualization
from .structures import DuckieGeometry


class DuckieGameVisualization(DrivingGameVisualization):
    def __init__(
            self,
            params,
            side: D,
            geometries: Mapping[PlayerName, DuckieGeometry],
            ds: D
    ):
        DrivingGameVisualization.__init__(
            self,
            params=params,
            side=side,
            geometries=geometries,
            ds=ds,
        )