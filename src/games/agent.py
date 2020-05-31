from decimal import Decimal as D

from possibilities import Poss, PossibilityStructure
from .game_def import AgentBelief, Dynamics, JointState, SR, U, X

__all__ = ["RandomAgent"]


class RandomAgent(AgentBelief[X, U]):
    dynamics: Dynamics[X, U, SR]
    ps: PossibilityStructure

    def __init__(self, dynamics: Dynamics[X, U, SR], ps: PossibilityStructure):
        self.dynamics = dynamics
        self.ps = ps

    def get_commands(self, state_self: X, state_others: Poss[JointState]) -> Poss[U]:
        options = self.dynamics.successors(state_self, dt=D(1))  # doesn't matter
        return self.ps.lift_many(options)
