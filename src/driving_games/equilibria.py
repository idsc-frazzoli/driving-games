import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Generic, Mapping, Set, TypeVar

from frozendict import frozendict

from zuper_commons.types import ZAssertionError, ZValueError
from .game_def import ASet, PlayerName
from .poset import COMP_OUTCOMES, ComparisonOutcome, FIRST_PREFERRED, Preference
from . import logger
from .poset_lexi import StrictProductPreference
from .poset_sets import remove_dominated

X = TypeVar("X")
O = TypeVar("O")

Choice = TypeVar("Choice")


@dataclass
class PointStats:
    happy: ASet[PlayerName]
    unhappy: ASet[PlayerName]
    outcome: O
    alternatives: Mapping[PlayerName, ASet[ComparisonOutcome]]


@dataclass
class EquilibriaAnalysis(Generic[Choice]):
    nondom_nash_equilibria: Mapping[Mapping[PlayerName, Choice], O]
    nash_equilibria: Mapping[Mapping[PlayerName, Choice], O]
    ps: Dict[Mapping[PlayerName, Choice], PointStats]


def analyze_equilibria(
    action2outcome: Mapping[Mapping[PlayerName, Choice], O],
    preferences: Mapping[PlayerName, Preference[O]],
) -> EquilibriaAnalysis:
    # we want to make sure that there are all combinations
    combos: Combos[Choice] = check_contains_all_combo(action2outcome)
    player_names = set(combos.player2choices)
    if set(preferences) != set(player_names):
        raise ZValueError(action2outcome=action2outcome,
                          preferences=preferences)
    # logger.info(combos=combos)
    comb: Mapping[PlayerName, Choice]
    ps: Dict[Mapping[PlayerName, Choice], PointStats] = {}
    x0: Mapping[PlayerName, Choice]
    x1: Mapping[PlayerName, Choice]

    nash_equilibria = {}
    action_to_change: Choice
    for x0 in combos.all_comb:
        assert x0 in action2outcome
        happy_players = set()
        unhappy_players = set()
        alternatives = {}
        for player_name in player_names:
            pref = preferences[player_name]
            is_happy = True
            variations_ = variations(combos, x0, player_name)
            # if not variations_:
            #     logger.info(combos=combos,x0=x0, player_name=player_name)
            # assert len(variations_) >= 1
            alternatives_player = {}
            # logger.info('looking for variations', variations_=variations_)
            for action_to_change, x1 in variations_.items():
                zassert(x1 in action2outcome, x1=x1, action2outcome=set(action2outcome))
                o1, o0 = action2outcome[x1], action2outcome[x0]
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
        stats = PointStats(happy=frozenset(happy_players), unhappy=frozenset(unhappy_players),
                           outcome=action2outcome[x0],
                           alternatives=frozendict(alternatives))
        ps[x0] = stats

        if not unhappy_players:
            nash_equilibria[x0] = stats.outcome
    # logger.info(ps=ps)

    # we need something to compare set of outcomes
    pref = StrictProductPreference(preferences)
    logger.info(nash_equilibria=nash_equilibria)
    nondom_nash_equilibria = remove_dominated(nash_equilibria, pref)

    return EquilibriaAnalysis(nondom_nash_equilibria=nondom_nash_equilibria,
                              nash_equilibria=nash_equilibria
                              , ps=ps)



def zassert(val: bool, **kwargs):
    if not val:
        msg = "Assertion failed"
        raise ZAssertionError(msg, val=val, **kwargs)


@dataclass
class Combos(Generic[Choice]):
    all_comb: ASet[Mapping[PlayerName, Choice]]
    player2choices: Mapping[PlayerName, Set[Choice]]


def variations(
    c: Combos[Choice], x0: Mapping[PlayerName, Choice], player_name: PlayerName
) -> Mapping[Choice, Mapping[PlayerName, Choice]]:
    all_actions = set(c.player2choices[player_name])
    current_action = x0[player_name]
    assert current_action in all_actions, (current_action, all_actions)
    all_actions.remove(current_action)

    # assert len(all_actions) >= 1, c.player2choices[player_name]
    res = {}
    for alternative in all_actions:
        d = dict(x0)
        d[player_name] = alternative
        res[alternative] = frozendict(d)
    return frozendict(res)


def check_contains_all_combo(
    action2outcome: Mapping[Mapping[PlayerName, Choice], ASet[O]]
) -> Combos[Choice]:
    player2choices: Dict[PlayerName, Set[X]] = defaultdict(set)
    for actions in action2outcome:
        for player_name, action in actions.items():
            player2choices[player_name].add(action)

    all_comb = get_all_combinations(player2choices)
    for c in all_comb:
        if c not in action2outcome:
            msg = "Missing combination"
            raise ZValueError(msg, c=c, action2outcome=list(action2outcome))
    return Combos(all_comb, player2choices)


def get_all_combinations(
    player2choices: Mapping[PlayerName, Set[X]]
) -> ASet[Mapping[PlayerName, X]]:
    players = list(player2choices)
    choices = list(player2choices[_] for _ in players)
    all_combs = itertools.product(*tuple(choices))
    res = set()
    for c in all_combs:
        comb = frozendict(zip(players, c))
        res.add(comb)
    return frozenset(res)
