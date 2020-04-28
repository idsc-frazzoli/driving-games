from dataclasses import dataclass
from typing import Dict

from driving_games.game_generation import get_game1
from games import Game

__all__ = ["games_zoo"]


@dataclass
class GameSpec:
    desc: str
    game: Game


games_zoo: Dict[str, GameSpec] = {}

# Creates the examples
games_zoo["game1"] = GameSpec(desc="", game=get_game1())
