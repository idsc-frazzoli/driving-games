import itertools
from collections import defaultdict
from typing import AbstractSet, Collection, Dict, FrozenSet, Iterable, Mapping, Set, TypeVar

from frozendict import frozendict

from possibilities import Poss, PossibilityStructure
from .game_def import (
    check_joint_mixed_actions2,
    check_joint_pure_actions,
    JointMixedActions2,
    JointPureActions,
    PlayerName,
    PlayerOptions,
    Pr,
    SetOfOutcomes,
    U,
)

__all__ = []


def mixed_from_pure(pure_actions: JointPureActions) -> JointMixedActions2:
    return frozendict({k: frozenset({v}) for k, v in pure_actions.items()})  # FIXME


def get_all_choices_by_players(possibile: Collection[JointPureActions]) -> PlayerOptions:
    player2choices: Dict[PlayerName, Set[U]] = defaultdict(set)
    for pure_actions in possibile:
        for player_name, u in pure_actions.items():
            # assert not isinstance(action, (frozenset, set)), action
            player2choices[player_name].add(u)
    res: Dict[PlayerName, FrozenSet[U]] = {}
    for player_name, player_actions in player2choices.items():
        res[player_name] = frozenset(player_actions)

    return frozendict(res)


def get_all_combinations(*, mixed_actions: JointMixedActions2) -> FrozenSet[JointPureActions]:
    check_joint_mixed_actions2(mixed_actions)
    players = list(mixed_actions)
    choices = list(mixed_actions[_] for _ in players)
    all_combs = itertools.product(*tuple(choices))
    res = set()
    for c in all_combs:
        comb = frozendict(zip(players, c))
        check_joint_pure_actions(comb)
        res.add(comb)
    r = frozenset(res)

    return r


def all_pure_actions(mixed_actions: JointMixedActions2) -> FrozenSet[JointPureActions]:
    check_joint_mixed_actions2(mixed_actions)
    names = list(mixed_actions)
    possible = [mixed_actions[_] for _ in names]
    res = []
    for combination in itertools.product(*tuple(possible)):
        pure_actions = frozendict(dict(zip(names, combination)))
        check_joint_pure_actions(pure_actions)
        res.append(pure_actions)
    ret = frozenset(res)
    return ret


def flatten_outcomes(
    solved: Mapping[JointPureActions, SetOfOutcomes], options: Poss[JointPureActions, Pr]
) -> SetOfOutcomes:
    return flatten_sets(solved[_] for _ in options)


M = TypeVar("M")


def flatten_sets(x: Iterable[AbstractSet[M]]) -> FrozenSet[M]:
    res = set()
    for _ in x:
        res.update(_)
    return frozenset(res)


def add_action(
    ps: PossibilityStructure[Pr],
    player_name: PlayerName,
    player_action: U,
    set_pure_actions_others: Poss[JointPureActions, Pr],
) -> Poss[JointPureActions, Pr]:
    def add(x: JointPureActions) -> JointPureActions:
        v2 = dict(x)
        v2[player_name] = player_action
        return v2

    return ps.build(set_pure_actions_others, add)
