from shapely.geometry import Polygon

from dg_commons.sim import CollisionReportPlayer
from driving_games import VehicleState, VehicleActions, VehicleObs, VehicleCosts
from games import GamePlayer, Game


class DrivingGame(Game[VehicleState, VehicleActions, VehicleObs, VehicleCosts, CollisionReportPlayer, Polygon]):
    pass


class DrivingGamePlayer(
    GamePlayer[VehicleState, VehicleActions, VehicleObs, VehicleCosts, CollisionReportPlayer, Polygon]
):
    pass
