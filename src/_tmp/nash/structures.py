from dataclasses import dataclass
from math import isclose
from typing import NewType, Tuple, Optional

import numpy as np

__all__ = ["Equilibrium", "BiMatGame", "PlayerType", "MAXIMIZER", "MINIMIZER"]


@dataclass(frozen=True, unsafe_hash=True)
class Equilibrium:
    s1: np.ndarray
    s2: np.ndarray
    p1_payoff: float
    p2_payoff: float

    def __eq__(self, other: "Equilibrium"):
        """This actually implements an ALMOST EQUAL test"""
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return (
            isclose(self.p1_payoff, other.p1_payoff)
            and isclose(self.p2_payoff, other.p2_payoff)
            and np.allclose(self.s1, other.s1)
            and np.allclose(self.s2, other.s2)
        )


PlayerType = NewType("PlayerType", str)
MAXIMIZER = PlayerType("max")
MINIMIZER = PlayerType("min")


@dataclass(frozen=True, unsafe_hash=True)
class BiMatGame:
    A: np.ndarray
    B: np.ndarray
    equilibria: Optional[Tuple[Equilibrium, ...]] = None
    p1_type: PlayerType = MINIMIZER
    p2_type: PlayerType = MINIMIZER
    desc: str = ""


def print_bimatgame(game: BiMatGame) -> str:
    rows, cols = game.A.shape
    str_game = ""
    for r in range(rows):
        str_game += "\n\t"
        for c in range(cols):
            str_game += "{},{}\t\t".format(game.A[r, c], game.B[r, c])
    return str_game
