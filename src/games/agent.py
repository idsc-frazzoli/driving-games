import random
from abc import ABC
from decimal import Decimal as D
from typing import Generic, Mapping

from .game_def import ASet, Dynamics, PlayerName, U, X

__all__ = ['RandomAgent']


class RandomAgent(Generic[X, U], ABC):
    dynamics: Dynamics[X, U]

    def __init__(self, dynamics: Dynamics[X, U]):
        self.dynamics = dynamics

    def get_commands(self, state_self: X, state_others: Mapping[PlayerName, ASet[X]]) -> ASet[U]:
        options = self.dynamics.successors(state_self, dt=D(1))  # doesn't matter
        possible = list(options)
        return random.choice(possible)
