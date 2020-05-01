import itertools
from typing import Dict, FrozenSet, List, Mapping, Tuple

from frozendict import frozendict
from toolz import keyfilter

from possibilities import check_poss, Poss, PossibilityStructure
from preferences import Preference, remove_dominated
from preferences.operations import worst_cases
from zuper_commons.types import ZValueError
from .equilibria import EquilibriaAnalysis
from .game_def import (
    check_joint_mixed_actions2,
    check_joint_pure_actions,
    JointMixedActions2,
    JointPureActions,
    Outcome,
    PlayerName,
    Pr,
    RJ,
    RP,
    SetOfOutcomes,
    U,
    X,
    Y,
)


def get_security_policies(
    ps: PossibilityStructure[Pr],
    solved: Mapping[JointPureActions, SetOfOutcomes],
    preferences: Mapping[PlayerName, Preference[SetOfOutcomes]],
    ea: EquilibriaAnalysis[Pr, X, U, Y, RP, RJ],
) -> JointMixedActions2:
    actions: Dict[PlayerName, Poss[U, Pr]] = {}
    for player_name in ea.player_mixed_strategies:
        player_pref = preferences[player_name]
        sp = get_security_policy(ps=ps, player_name=player_name, preference=player_pref, ea=ea, solved=solved)
        # check_poss(sp)
        # for _ in sp.support():
        #     assert not isinstance(_, Poss), sp
        actions[player_name] = sp

    return frozendict(actions)


def get_security_policy(
    ps: PossibilityStructure[Pr],
    solved: Mapping[JointPureActions, SetOfOutcomes],
    # moves: JointMixedActions2,
    player_name: PlayerName,
    preference: Preference[SetOfOutcomes],
    ea: EquilibriaAnalysis[Pr, X, U, Y, RP, RJ],
) -> Poss[U, Pr]:
    player_choices = ea.player_mixed_strategies[player_name]
    others_choices = frozendict(keyfilter(lambda _: _ != player_name, ea.player_mixed_strategies))
    # preferences: Dict[PlayerName, Preference[SetOfOutcomes]]

    action2outcomes: Dict[U, Poss[Outcome, Pr]] = {}
    player_choice: Poss[U, Pr]
    for player_choice in player_choices:
        option_outcomes = what_if_player_chooses(
            ps,
            ea=ea,
            player_name=player_name,
            player_action=player_choice,
            others_choices=others_choices,
            solved=solved,
            preference=preference,
        )
        action2outcomes[player_choice] = option_outcomes

    plausible = remove_dominated(action2outcomes, preference)
    ret = ps.lift_many(plausible)
    ret = ps.flatten(ret)
    return ret
    # player_could_do: ASet[U]  =


def what_if_player_chooses(
    ps: PossibilityStructure[Pr],
    player_name: PlayerName,
    ea: EquilibriaAnalysis[Pr, X, U, Y, RP, RJ],
    solved: Mapping[JointPureActions, SetOfOutcomes],
    player_action: Poss[U, Pr],
    others_choices: Mapping[PlayerName, FrozenSet[Poss[U, Pr]]],
    preference: Preference[SetOfOutcomes],
) -> SetOfOutcomes:
    """ Assume the player chooses u, and the others choose any other mixed policy.
        What is the worst case? """
    assert player_name not in others_choices
    # I have decided to do player_action
    # While I assume the others are going to mix theirs
    choices = dict(others_choices)
    choices[player_name] = frozenset({player_action})

    mixed: Mapping[Mapping[PlayerName, Poss[U, Pr]], SetOfOutcomes]
    mixed = get_mixed(ps, choices, solved)

    w = worst_cases(mixed, preference)
    # Note that there might be more nondonimnate
    values = list(w.values())
    # XXX not sure it is so simple
    return ps.flatten(ps.lift_many(values))


def get_mixed(
    ps: PossibilityStructure[Pr],
    choices: Mapping[PlayerName, FrozenSet[Poss[U, Pr]]],
    pure_outcomes: Mapping[JointPureActions, SetOfOutcomes],
) -> Mapping[Mapping[PlayerName, Poss[U, Pr]], SetOfOutcomes]:
    players_ordered = list(choices)
    players_strategies = [choices[_] for _ in players_ordered]
    results: Dict[Mapping[PlayerName, Poss[U, Pr]], SetOfOutcomes] = {}
    for choices in itertools.product(*tuple(players_strategies)):
        choice: Mapping[PlayerName, Poss[U, Pr]] = frozendict(zip(players_ordered, choices))
        #
        # p: List[Tuple[JointPureActions, Pr]] = []
        # choose: List[FrozenSet[U]] = [choice[k].support() for k in players_ordered]
        # for pure in itertools.product(*tuple(choose)):
        #     pure_action: JointPureActions = frozendict(zip(players_ordered, pure))
        #     probs: Tuple = tuple(
        #         choice[player_name].get(pure_action[player_name])
        #         for player_name in players_ordered
        #     )
        #     p.append((pure_action, ps.multiply(probs)))
        # dist: Poss[JointPureActions, Pr] = ps.fold(p)
        dist: Poss[JointPureActions, Pr]
        dist = get_mixed2(ps, choice)
        mixed_outcome: Poss[SetOfOutcomes, Pr] = ps.build(dist, pure_outcomes.__getitem__)
        results[choice] = ps.flatten(mixed_outcome)
    return results


def get_mixed2(
    ps: PossibilityStructure[Pr], mixed: Mapping[PlayerName, Poss[U, Pr]]
) -> Poss[JointPureActions, Pr]:
    check_joint_mixed_actions2(mixed)
    for k, v in mixed.items():
        check_poss(v)
        for x in v.support():
            if isinstance(x, Poss):
                raise ZValueError(x=x, v=v, k=k, mixed=mixed)

    # p: List[Tuple[JointPureActions, Pr]] = []
    # players_ordered = list(mixed)
    # choose: List[FrozenSet[U]] = [mixed[k].support() for k in players_ordered]
    # for pure in itertools.product(*tuple(choose)):
    #     pure_action: JointPureActions = frozendict(zip(players_ordered, pure))
    #     check_joint_pure_actions(pure_action, pure=pure, choose=choose, mixed=mixed)
    #     probs: Tuple = tuple(
    #         mixed[player_name].get(pure_action[player_name]) for player_name in players_ordered
    #     )
    #     p.append((pure_action, ps.multiply(probs)))
    # dist: Poss[JointPureActions, Pr] = ps.fold(p)

    def f(y: JointPureActions) -> JointPureActions:
        return y

    dist: Poss[JointPureActions, Pr] = ps.build_multiple(a=mixed, f=f)

    for _ in dist.support():
        check_joint_pure_actions(_)
    return dist


#
# def add_action(
#     ps: PossibilityStructure[Pr],
#     player_name: PlayerName,
#     player_action: U,
#     set_pure_actions_others: Poss[JointPureActions, Pr],
# ) -> Poss[JointPureActions, Pr]:
#     def add(x: JointPureActions) -> JointPureActions:
#         v2 = dict(x)
#         v2[player_name] = player_action
#         return v2
#
#     return ps.build(set_pure_actions_others, add)
