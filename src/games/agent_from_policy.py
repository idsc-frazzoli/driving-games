from typing import Mapping, Optional

from frozendict import frozendict

from possibilities import Poss, PossibilityMonad
from zuper_commons.types import ZException, ZNotImplementedError, ZValueError
from .game_def import AgentBelief, JointState, U, X, PlayerName
from games import logger
from games.utils import fkeyfilter
from games.solve.solution_structures import GameFactorization

__all__ = ["DoesNotKnowPolicy", "AgentFromPolicy"]


class DoesNotKnowPolicy(ZException):
    pass


class AgentFromPolicy(AgentBelief[X, U]):
    policy: Mapping[X, Mapping[Poss[JointState], Poss[U]]]
    ps: PossibilityMonad
    gf: Optional[GameFactorization]
    player_name: Optional[PlayerName]

    def __init__(
        self,
        ps: PossibilityMonad,
        policy: Mapping[X, Mapping[Poss[JointState], Poss[U]]],
        gf: Optional[GameFactorization] = None,
        player_name: Optional[PlayerName] = None
    ):
        self.policy = policy
        self.ps = ps
        self.gf = gf
        self.player_name = player_name

    def get_commands(self, state_self: X, state_others: Poss[JointState]) -> Poss[U]:
        if state_self not in self.policy:
            msg = "I do not know the policy for this state"
            raise DoesNotKnowPolicy(
                msg,
                state_self=state_self,
                state_others=state_others,
                states_self_known=set(self.policy),
            )

        lookup = self.policy[state_self]

        if state_others in lookup:
            return lookup[state_others]
        else:

            # when_nobody_there = self.ps.unit(frozendict())
            # if when_nobody_there in lookup:
            #     return lookup[when_nobody_there]

            if self.gf:
                reduced_state_others = self._get_reduced_state_others(state_self, state_others)
                if reduced_state_others in lookup:
                    return lookup[reduced_state_others]
                raise ZNotImplementedError(state_self=state_self, state_others=state_others, lookup=lookup,
                                           reduced_state_others=reduced_state_others)
            else:
                if len(lookup) == 1:  # for the two player ghost tree
                    logger.warning(
                        "Some players are ignored in simultaion! Should only happen in sequential play"
                    )
                    return list(lookup.values())[0]

            raise ZNotImplementedError(state_self=state_self, state_others=state_others, lookup=lookup)

    def _get_reduced_state_others(
            self,
            state_self: X,
            state_others: Poss[JointState]
    ) -> Poss[JointState]:
        """
        Get a reduced state_others for which the players not relevant (found in game factorization) are removed.

        :param state_self: The agents own state
        :param state_others: Distribution over the other players possible states
        """

        def reduce_joint_states(js_others: JointState) -> JointState:
            js = frozendict({self.player_name: state_self, **js_others})  # Get the full joint state
            if js not in self.gf.ipartitions:
                raise ZValueError("Cannot find the state in the factorization info")
            partitions = self.gf.ipartitions[js]  # get the partitions for this joint state
            for players_in_partition in partitions:
                if self.player_name in players_in_partition:
                    other_players_partition = players_in_partition - {self.player_name, }  # remove the agent from set
                    # get only the relevant states by ignoring the players not relevant
                    js_others_reduced = fkeyfilter(lambda pn: pn in other_players_partition, js_others)
                    return js_others_reduced
            raise ZValueError("Could not reduce joint state")

        # remove all players that the agent can ignore at this state
        state_others_reduced = self.ps.build(state_others, reduce_joint_states)

        return state_others_reduced
