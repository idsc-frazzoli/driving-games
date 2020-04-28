import itertools
from collections import defaultdict
from typing import Collection, Dict, Set

from frozendict import frozendict

from .game_def import (
    ASet,
    check_joint_mixed_actions,
    check_joint_pure_actions,
    JointMixedActions,
    JointPureActions,
    PlayerName,
    U,
)

__all__ = []


def mixed_from_pure(pure_actions: JointPureActions) -> JointMixedActions:
    return frozendict({k: frozenset({v}) for k, v in pure_actions.items()})


def get_all_choices_by_players(possibile: Collection[JointPureActions]) -> JointMixedActions:
    player2choices: Dict[PlayerName, Set[U]] = defaultdict(set)
    for pure_actions in possibile:
        for player_name, u in pure_actions.items():
            # assert not isinstance(action, (frozenset, set)), action
            player2choices[player_name].add(u)
    res: Dict[PlayerName, ASet[U]] = {}
    for player_name, player_actions in player2choices.items():
        res[player_name] = frozenset(player_actions)

    return frozendict(res)


def get_all_combinations(*, mixed_actions: JointMixedActions) -> ASet[JointPureActions]:
    check_joint_mixed_actions(mixed_actions)
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


def all_pure_actions(mixed_actions: JointMixedActions) -> ASet[JointPureActions]:
    check_joint_mixed_actions(mixed_actions)
    names = list(mixed_actions)
    possible = [mixed_actions[_] for _ in names]
    res = []
    for combination in itertools.product(*tuple(possible)):
        pure_actions = frozendict(dict(zip(names, combination)))
        check_joint_pure_actions(pure_actions)
        res.append(pure_actions)
    ret = frozenset(res)
    return ret
