from dataclasses import dataclass

from bayesian_driving_games.structures import BayesianGame
from games import Game

__all__ = ["GameSpec"]


@dataclass
class GameSpec:
    desc: str
    game: Game


@dataclass
class BayesianGameSpec(GameSpec):
    game: BayesianGame
