from dataclasses import dataclass

from games import Game


@dataclass
class GameSpec:
    desc: str
    game: Game
