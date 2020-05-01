import itertools
from dataclasses import dataclass
from typing import Dict, FrozenSet, Generic, List, Mapping, Set, Tuple

from frozendict import frozendict
from . import logger
from possibilities import Poss, PossibilityStructure
from preferences import (
    COMP_OUTCOMES,
    ComparisonOutcome,
    FIRST_PREFERRED,
    Preference,
    remove_dominated,
    StrictProductPreference,
)
from zuper_commons.types import ZAssertionError
from . import GameConstants
from .comb_utils import valmap
from .game_def import (
    check_joint_mixed_actions2,
    check_set_outcomes,
    JointMixedActions2,
    JointPureActions,
    PlayerName,
    PlayerOptions,
    Pr,
    RJ,
    RP,
    SetOfOutcomes,
    U,
    X,
    Y,
)

__all__ = []


# Choice = TypeVar("Choice")
# Choice = ASet[U]
# Mapping[PlayerName, Choice] -> JointMixedActions
# O -> SetOfOutcomes


@dataclass
class PointStats(Generic[Pr, X, U, Y, RP, RJ]):
    happy: FrozenSet[PlayerName]
    unhappy: FrozenSet[PlayerName]
    outcome: SetOfOutcomes
    alternatives: Mapping[PlayerName, FrozenSet[ComparisonOutcome]]

    def __post_init__(self):
        if not GameConstants.checks:
            return

        check_set_outcomes(self.outcome)


@dataclass
class EquilibriaAnalysis(Generic[Pr, X, U, Y, RP, RJ]):
    player_mixed_strategies: Mapping[PlayerName, FrozenSet[Poss[U, Pr]]]
    nondom_nash_equilibria: Mapping[JointMixedActions2, SetOfOutcomes]
    nash_equilibria: Mapping[JointMixedActions2, SetOfOutcomes]
    ps: Dict[JointMixedActions2, PointStats]

    def __post_init__(self):
        if not GameConstants.checks:
            return

        for _ in self.ps:
            check_joint_mixed_actions2(_)
        for _ in self.nondom_nash_equilibria:
            check_joint_mixed_actions2(_)
        for _ in self.nash_equilibria:
            check_joint_mixed_actions2(_)


# @dataclass
# class Combos(Generic[Pr, X, U, Y, RP, RJ]):
#     all_comb: FrozenSet[JointPureActions]
#     player2choices: JointMixedActions2
#
#     def __post_init__(self):
#         if not GameConstants.checks:
#             return
#
#         check_isinstance(self.all_comb, frozenset)
#         for _ in self.all_comb:
#             check_joint_pure_actions(_)


def analyze_equilibria(
    *,
    ps: PossibilityStructure[Pr],
    moves: PlayerOptions,
    solved: Mapping[JointPureActions, SetOfOutcomes],
    preferences: Mapping[PlayerName, Preference[SetOfOutcomes]],
) -> EquilibriaAnalysis:
    # First we want to make sure that there are all combinations
    # combos: Combos[Pr, X, U, Y, RP, RJ] = check_contains_all_combo(frozenset(solved))
    # player_names = set(combos.player2choices)
    # if set(preferences) != set(player_names):  # pragma: no cover
    #     raise ZValueError(solved=solved, preferences=preferences)

    # Now we want to find all mixed strategies
    # Example: From sets, you could have [A, B] ->  {A}, {B}, {A,B}
    # Example: From probs, you could have [A,B] -> {A:1}, {B:1} , {A:0.5, B:0.5}, ...

    player_mixed_strategies: Dict[PlayerName, FrozenSet[Poss[U, Pr]]] = valmap(ps.mix, moves)
    # logger.info(player_mixed_strategies=player_mixed_strategies)
    # now we do the product of the mixed strategies
    # let's order them
    players_ordered = list(player_mixed_strategies)
    players_strategies = [player_mixed_strategies[_] for _ in players_ordered]

    results: Dict[Mapping[PlayerName, Poss[U, Pr]], SetOfOutcomes] = {}
    for choices in itertools.product(*tuple(players_strategies)):
        choice: Mapping[PlayerName, Poss[U, Pr]] = frozendict(zip(players_ordered, choices))
        #
        # p: List[Tuple[JointPureActions, Pr]] = []
        # choose: List[FrozenSet[U]] = [choice[k].support() for k in players_ordered]
        # for pure in itertools.product(*tuple(choose)):
        #     pure_action: JointPureActions = frozendict(zip(players_ordered, pure))
        #     probs: Tuple = tuple(
        #         choice[player_name].get(pure_action[player_name]) for player_name in players_ordered
        #     )
        #
        #     p.append((pure_action, ps.multiply(probs)))
        # dist: Poss[JointPureActions, Pr] = ps.fold(p)
        #
        def f(y: JointPureActions) -> JointPureActions:
            return y

        dist: Poss[JointPureActions, Pr] = ps.build_multiple(a=choice, f=f)
        # logger.info(dist=dist,
        #             solved=solved)
        mixed_outcome: Poss[SetOfOutcomes, Pr] = ps.build(dist, solved.__getitem__)
        results[choice] = ps.flatten(mixed_outcome)

    # logger.info(results=results)

    return analyze(player_mixed_strategies, results, preferences)


def analyze(
    player_mixed_strategies: Mapping[PlayerName, FrozenSet[Poss[U, Pr]]],
    results: Mapping[Mapping[PlayerName, Poss[U, Pr]], SetOfOutcomes],
    preferences: Mapping[PlayerName, Preference[SetOfOutcomes]],
):
    # logger.info(combos=combos)
    comb: JointPureActions
    ps: Dict[JointPureActions, PointStats] = {}
    x0: Mapping[PlayerName, Poss[U, Pr]]
    x1: Mapping[PlayerName, Poss[U, Pr]]
    player_names = set(player_mixed_strategies)
    nash_equilibria = {}
    action_to_change: FrozenSet[U]
    for x0 in results:

        happy_players = set()
        unhappy_players = set()
        alternatives = {}
        for player_name in player_names:
            pref = preferences[player_name]
            is_happy: bool = True
            variations_: Mapping[U, Mapping[PlayerName, Poss[U, Pr]]] = variations(
                player_mixed_strategies, x0, player_name
            )
            # if not variations_:
            #     logger.info(combos=combos,x0=x0, player_name=player_name)
            # assert len(variations_) >= 1
            alternatives_player = {}
            # logger.info('looking for variations', variations_=variations_)
            for action_to_change, x1 in variations_.items():
                # zassert(x1 in results, x1=x1, results=set(results))
                o1, o0 = results[x1], results[x0]
                res = pref.compare(o1, o0)
                assert res in COMP_OUTCOMES, (res, pref)
                # logger.info(o1=o1, o0=o0, res=res)
                if res == FIRST_PREFERRED:
                    is_happy = False
                alternatives_player[action_to_change] = res
            alternatives[player_name] = frozendict(alternatives_player)
            if is_happy:
                happy_players.add(player_name)
            else:
                unhappy_players.add(player_name)
        stats = PointStats(
            happy=frozenset(happy_players),
            unhappy=frozenset(unhappy_players),
            outcome=results[x0],
            alternatives=frozendict(alternatives),
        )
        ps[x0] = stats

        if not unhappy_players:
            nash_equilibria[x0] = stats.outcome
    # logger.info(ps=ps)

    # we need something to compare set of outcomes

    preferences_: Tuple[Preference[SetOfOutcomes], ...] = tuple(preferences.values())
    pref: Preference[SetOfOutcomes] = StrictProductPreference(preferences_)

    # logger.info(nash_equilibria=nash_equilibria, preferences=preferences, pref=pref)
    nondom_nash_equilibria = remove_dominated(nash_equilibria, pref)

    return EquilibriaAnalysis(
        player_mixed_strategies=player_mixed_strategies,
        nondom_nash_equilibria=nondom_nash_equilibria,
        nash_equilibria=nash_equilibria,
        ps=ps,
    )


def zassert(val: bool, **kwargs):
    if not val:  # pragma: no cover
        msg = "Assertion failed"
        raise ZAssertionError(msg, val=val, **kwargs)


def variations(
    player_mixed_strategies: Mapping[PlayerName, FrozenSet[Poss[U, Pr]]],
    x0: Mapping[PlayerName, Poss[U, Pr]],
    player_name: PlayerName,
) -> Mapping[U, Mapping[PlayerName, Poss[U, Pr]]]:
    # check_joint_pure_actions(x0)
    all_mixed_actions: Set[Poss[U, Pr]] = set(player_mixed_strategies[player_name])
    current_action: Poss[U, Pr] = x0[player_name]
    assert current_action in all_mixed_actions, (current_action, all_mixed_actions)
    all_mixed_actions.remove(current_action)

    # assert len(all_actions) >= 1, c.player2choices[player_name]
    res = {}
    for alternative in all_mixed_actions:
        d = dict(x0)
        d[player_name] = alternative
        _ = frozendict(d)
        # check_joint_pure_actions(_)
        res[alternative] = _
    return frozendict(res)
