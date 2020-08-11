from typing import Iterable, Tuple, NewType

import numpy as np
from dataclasses import dataclass

from nash.structures import Equilibrium

__all__ = ["BiMatGame", "matching_pennies", "matching_pennies_2"]

PlayerType = NewType("PlayerType", str)
MAXIMIZER = PlayerType("max")
MINIMIZER = PlayerType("min")


@dataclass(frozen=True, unsafe_hash=True)
class BiMatGame:
    desc: str
    p1_type: PlayerType
    p2_type: PlayerType
    A: np.ndarray
    B: np.ndarray
    equilibria: Tuple[Equilibrium, ...]


matching_pennies = BiMatGame(
    desc="Standard matching pennies game, players are maximizers",
    p1_type=MAXIMIZER, p2_type=MAXIMIZER,
    A=np.array([[1, -1], [-1, 1]]),
    B=np.array([[-1, 1], [1, -1]]),
    equilibria=(Equilibrium(s1=np.array([0.5, 0.5]), s2=np.array([0.5, 0.5]), p1_payoff=0, p2_payoff=0),)
)

matching_pennies_2 = BiMatGame(
    desc="Variation of the matching pennies, players are maximizers",
    p1_type=MAXIMIZER, p2_type=MAXIMIZER,
    A=np.array([[7, -1], [-1, 1]]),
    B=np.array([[-1, 1], [1, -1]]),
    equilibria=(Equilibrium(s1=np.array([.5, .5]), s2=np.array([.2, .8]), p1_payoff=7*.2-.8, p2_payoff=0),)
)

degenerate_1 = BiMatGame(
    desc="""Degenerate game with 8 NE. The game defined by
        [0, 0, 6, 0, 0]         [3, 0, 2, 1, 0]
    A = [0, 0, 3, 2, 1],    B = [0, 2, 0, 0, 4]
        [4, 3, 0, 0, 1]         [4, 0, 2, 4, 4]
    is supposed to have 8 NE:
y: [ 1 0 0 ]        J1: 0       z: [ 0  1   0   0   0 ]        J2: 0
y: [ 1 0 0 ]        J1: 0       z: [ 0  0   0   0   1 ]        J2: 0
y: [ 2/3 1/3 0 ]    J1: 0       z: [ 0  1   0   0   0 ]        J2: 2/3
y: [ 1/3 1/2 1/6 ]  J1: 4/3     z: [ 0  4/9 2/9 1/3 0 ]        J2: 1
y: [ 0 1/2 1/2 ]    J1: 3/2     z: [ 0  1/2 1/2 0   0 ]        J2: 1
y: [ 0 1 0 ]        J1: 12/7    z: [3/7 0   4/7 0   0 ]        J2: 0
y: [ 0 1 0 ]        J1: 0       z: [1   0   0   0   0 ]        J2: 0
y: [ 0 1 0 ]        J1: 3/2     z: [3/8 0   1/4 3/8 0 ]        J2: 0.
    Players are minimizers. """,
    p1_type=MINIMIZER, p2_type=MINIMIZER,
    A=np.array([[0, 0, 6, 0, 0], [0, 0, 3, 2, 1], [4, 3, 0, 0, 1]]),
    B=np.array([[3, 0, 2, 1, 0], [0, 2, 0, 0, 4], [4, 0, 2, 4, 4]]),
    equilibria=(
        Equilibrium(s1=np.array([1, 0, 0]), s2=np.array([0, 1, 0, 0, 0]), p1_payoff=0, p2_payoff=0),
        Equilibrium(s1=np.array([1, 0, 0]), s2=np.array([0, 0, 0, 0, 1]), p1_payoff=0, p2_payoff=0),
        Equilibrium(s1=np.array([2/3, 1/3, 0]), s2=np.array([0, 1, 0, 0, 0]), p1_payoff=0, p2_payoff=2/3),
        Equilibrium(s1=np.array([1/3, 1/2, 1/6]), s2=np.array([0, 4/9, 2/9, 1/3, 0]), p1_payoff=4/3,
                    p2_payoff=1),
        Equilibrium(s1=np.array([0, 1/2, 1/2]), s2=np.array([0, 1/2, 1/2, 0, 0]), p1_payoff=3/2, p2_payoff=1),
        Equilibrium(s1=np.array([0, 1, 0]), s2=np.array([3/7, 0, 4/7, 0, 0]), p1_payoff=12/7, p2_payoff=0),
        Equilibrium(s1=np.array([0, 1, 0]), s2=np.array([1, 0, 0, 0, 0]), p1_payoff=0, p2_payoff=0),
        Equilibrium(s1=np.array([0, 1, 0]), s2=np.array([0, 1, 0, 0, 0]), p1_payoff=3/2, p2_payoff=0),
    )
)
