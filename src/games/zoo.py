from dataclasses import dataclass
from games import Game

__all__ = ["GameSpec"]


@dataclass
class GameSpec:
    desc: str
    game: Game
