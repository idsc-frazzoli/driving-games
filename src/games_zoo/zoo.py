from typing import Dict

from driving_games import driving_games_zoo
from games import GameSpec

__all__ = ["games_zoo"]

games_zoo: Dict[str, GameSpec] = {}

games_zoo.update(driving_games_zoo)
