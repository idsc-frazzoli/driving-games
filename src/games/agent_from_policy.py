from typing import Mapping

from games import JointState, U, X
from games.game_def import AgentBelief, Pr
from possibilities import Poss
from zuper_commons.types import ZException, ZNotImplementedError


class DoesNotKnowPolicy(ZException):
    pass


class AgentFromPolicy(AgentBelief[Pr, X, U]):
    policy: Mapping[X, Mapping[Poss[JointState, Pr], Poss[U, Pr]]]

    def __init__(self, policy: Mapping[X, Mapping[Poss[JointState, Pr], Poss[U, Pr]]]):
        self.policy = policy

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
            raise ZNotImplementedError(state_self=state_self, state_others=state_others, lookup=lookup)
