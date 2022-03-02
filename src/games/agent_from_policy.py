from functools import partial
from itertools import combinations, chain
from typing import Mapping, AbstractSet, List

from frozendict import frozendict
from zuper_commons.types import ZException

from dg_commons import PlayerName, U, X
from dg_commons.utils_toolz import V, K
from possibilities import Poss, PossibilityMonad
from .game_def import AgentBelief, JointState

__all__ = ["DoesNotKnowPolicy", "AgentFromPolicy"]


class DoesNotKnowPolicy(ZException):
    pass


class AgentFromPolicy(AgentBelief[X, U]):
    policy: Mapping[X, Mapping[Poss[JointState], Poss[U]]]
    ps: PossibilityMonad

    def __init__(
        self,
        ps: PossibilityMonad,
        policy: Mapping[X, Mapping[Poss[JointState], Poss[U]]],
        player_name: PlayerName,
    ):
        self.policy = policy
        self.ps = ps
        self.my_name = player_name

    def get_commands(self, state_self: X, state_others: Poss[JointState]) -> Poss[U]:
        if state_self not in self.policy:
            msg = "I do not know the policy for this state"
            # fixme does not find policy if the state is collided
            raise DoesNotKnowPolicy(
                msg,
                player_name=self.my_name,
                state_self=state_self,
                state_others=state_others,
                states_self_known=set(self.policy),
            )

        lookup: Mapping[Poss[JointState], Poss[U]] = self.policy[state_self]
        if len(lookup) == 1:
            return list(lookup.values())[0]

        if state_others in lookup:
            return lookup[state_others]
        else:
            # todo fixme for building factorization while creating the game tree
            # todo does it exist a subset of state of others that is present in the lookup?

            fact_state_of_others = self.find_candidate_factorized_states(state_others, lookup=lookup)
            return lookup[fact_state_of_others]

            # when_nobody_there = self.ps.unit(frozendict())
            # if when_nobody_there in lookup:
            #     return lookup[when_nobody_there]

        #    raise ZNotImplementedError(state_self=state_self, state_others=state_others, lookup=lookup)

    def find_candidate_factorized_states(
        self, state_others: Poss[JointState], lookup: Mapping[Poss[JointState], Poss[U]]
    ) -> Poss[JointState]:
        """
        Get a reduced state_others for which the players not relevant (found in game factorization) are
        removed.
        :param state_others: Distribution over the other players possible states
        """

        # fixme not sure this works if the state of other is not made only by ps.unit

        def gen_candidates(state_others_: JointState) -> Poss[Poss[JointState]]:
            power_js: AbstractSet[JointState] = set(fpowerdict(state_others_))
            return self.ps.lift_many([self.ps.unit(js_) for js_ in power_js])

        candidates: Poss[Poss[JointState]] = self.ps.join(self.ps.build(state_others, gen_candidates))
        found: List[Poss[JointState]] = [js for js in candidates.support() if js in lookup]
        if len(found) == 1:
            return found[0]
        elif len(found) > 1:
            # todo verify this hypotesis, shall we consider the match with the most players still active?
            return max(found, key=lambda x: len(next(iter(x.support()))))  # max(found, key=lambda x: len(x.support()))
        else:
            raise DoesNotKnowPolicy(
                state_others=state_others,
                lookup=lookup,
                candidates=candidates,
                found=found,
                len_found=len(found),
            )


def fpowerdict(dict_: Mapping[K, V]):
    return map(frozendict, powerset(dict_.items()))


def powerset(set_):
    return chain.from_iterable(map(partial(combinations, set_), range(len(set_) + 1)))
