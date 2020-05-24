from decimal import Decimal as D

from possibilities import Poss, PossibilityStructure
from .game_def import AgentBelief, Dynamics, JointState, Pr, SR, U, X

__all__ = ["RandomAgent"]


class RandomAgent(AgentBelief[Pr, X, U]):
    dynamics: Dynamics[Pr, X, U, SR]
    ps: PossibilityStructure[Pr]

    def __init__(self, dynamics: Dynamics[Pr, X, U, SR], ps: PossibilityStructure[Pr]):
        self.dynamics = dynamics
        self.ps = ps

    def get_commands(self, state_self: X, state_others: Poss[JointState, Pr]) -> Poss[U, Pr]:
        options = self.dynamics.successors(state_self, dt=D(1))  # doesn't matter
        return self.ps.lift_many(options)
