from typing import Mapping

from frozendict import frozendict

from possibilities import Poss, PossibilityStructure
from zuper_commons.types import ZException, ZNotImplementedError
from .game_def import AgentBelief, JointState, Pr, U, X

__all__ = ["DoesNotKnowPolicy", "AgentFromPolicy"]


class DoesNotKnowPolicy(ZException):
    pass


class AgentFromPolicy(AgentBelief[Pr, X, U]):
    policy: Mapping[X, Mapping[Poss[JointState, Pr], Poss[U, Pr]]]
    ps: PossibilityStructure[Pr]

    def __init__(
        self, ps: PossibilityStructure[Pr], policy: Mapping[X, Mapping[Poss[JointState, Pr], Poss[U, Pr]]]
    ):
        self.policy = policy
        self.ps = ps

    def get_commands(self, state_self: X, state_others: Poss[JointState, Pr]) -> Poss[U, Pr]:
        if state_self not in self.policy:
            msg = "I do not know the policy for this state"
            raise DoesNotKnowPolicy(
                msg, state_self=state_self, state_others=state_others, states_self_known=set(self.policy),
            )

        lookup = self.policy[state_self]
        if len(lookup) == 1:
            return list(lookup.values())[0]

        if state_others in lookup:
            return lookup[state_others]
        else:
            # SetPoss(_p=f❨f{}❩):

            when_nobody_there = self.ps.lift_one(frozendict())
            if when_nobody_there in lookup:
                return lookup[when_nobody_there]

            raise ZNotImplementedError(state_self=state_self, state_others=state_others, lookup=lookup)
