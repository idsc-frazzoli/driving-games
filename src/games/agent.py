from decimal import Decimal as D

from .game_def import AgentBelief, ASet, Dynamics, JointState, U, X

__all__ = ["RandomAgent"]


class RandomAgent(AgentBelief[X, U]):
    dynamics: Dynamics[X, U]

    def __init__(self, dynamics: Dynamics[X, U]):
        self.dynamics = dynamics

    def get_commands(self, state_self: X, state_others: ASet[JointState]) -> ASet[U]:
        options = self.dynamics.successors(state_self, dt=D(1))  # doesn't matter
        possible = list(options)
        return frozenset(possible)
