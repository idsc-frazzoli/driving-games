from shapely.geometry import Polygon

from dg_commons.sim import CollisionReportPlayer
from driving_games import VehicleTrackState, VehicleActions, VehicleObs, VehicleCosts
from games import GamePlayer, Game


class DrivingGame(Game[VehicleTrackState, VehicleActions, VehicleObs, VehicleCosts, CollisionReportPlayer, Polygon]):
    pass


class DrivingGamePlayer(
    GamePlayer[VehicleTrackState, VehicleActions, VehicleObs, VehicleCosts, CollisionReportPlayer, Polygon]
):
    pass
