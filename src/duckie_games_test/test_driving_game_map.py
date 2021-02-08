from decimal import Decimal as D

from duckie_games.rectangle import DrivingGameMap
from world.map_loading import load_driving_game_map
from games import logger


def test_driving_game_map():
    duckie_map = load_driving_game_map("4way-double")
    resource_cell_size = D(1)
    driving_game_map = DrivingGameMap.initializor(duckie_map=duckie_map, cell_size=resource_cell_size)

    logger.info(driving_game_map=driving_game_map)
