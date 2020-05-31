import itertools
from typing import Dict, FrozenSet, List, Mapping

from frozendict import frozendict
from toolz import keyfilter

from possibilities import check_poss, Poss, PossibilityStructure
from preferences import Preference, remove_dominated, worst_cases
from zuper_commons.types import ZValueError
from .equilibria import EquilibriaAnalysis
from .game_def import (
    check_joint_mixed_actions2,
    check_joint_pure_actions,
    JointMixedActions,
    JointPureActions,
    PlayerName,
    Pr,
    RJ,
    RP,
    U,
    UncertainCombined,
    X,
    Y,
)


def get_security_policies(
    ps: PossibilityStructure[Pr],
    solved: Mapping[JointPureActions, Mapping[PlayerName, UncertainCombined]],
    preferences: Mapping[PlayerName, Preference[UncertainCombined]],
    ea: EquilibriaAnalysis[Pr, X, U, Y, RP, RJ],
) -> JointMixedActions:
    actions: Dict[PlayerName, Poss[U, Pr]] = {}
    for player_name in ea.player_mixed_strategies:
        player_pref = preferences[player_name]
        sp = get_security_policy(ps=ps, player_name=player_name, preference=player_pref, ea=ea, solved=solved)

        actions[player_name] = sp

    return frozendict(actions)


def get_security_policy(
    ps: PossibilityStructure[Pr],
    solved: Mapping[JointPureActions, Mapping[PlayerName, UncertainCombined]],
    player_name: PlayerName,
    preference: Preference[UncertainCombined],
    ea: EquilibriaAnalysis[Pr, X, U, Y, RP, RJ],
) -> Poss[U, Pr]:
    player_choices = ea.player_mixed_strategies[player_name]
    others_choices = frozendict(keyfilter(lambda _: _ != player_name, ea.player_mixed_strategies))

    action2outcomes: Dict[Poss[U, Pr], UncertainCombined] = {}
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


def what_if_player_chooses(
    ps: PossibilityStructure[Pr],
    player_name: PlayerName,
    ea: EquilibriaAnalysis[Pr, X, U, Y, RP, RJ],
    solved: Mapping[JointPureActions, Mapping[PlayerName, UncertainCombined]],
    player_action: Poss[U, Pr],
    others_choices: Mapping[PlayerName, FrozenSet[Poss[U, Pr]]],
    preference: Preference[UncertainCombined],
) -> UncertainCombined:
    """
        Assume the player chooses u, and the others choose any other mixed policy.
        What is the worst case?

        :param ps: Possibility monad.
        :param player_name: Player name.

    """
    assert player_name not in others_choices
    # I have decided to do player_action
    # While I assume the others are going to mix theirs
    choices = dict(others_choices)
    choices[player_name] = frozenset({player_action})

    mixed: Mapping[JointMixedActions, UncertainCombined]
    mixed = _what_if_player_chooses_get_mixed(ps, choices, solved, player_name)

    w: Mapping[JointMixedActions, UncertainCombined]
    w = worst_cases(mixed, preference)
    # Note that there might be more nondominated
    values: List[UncertainCombined]
    values = list(w.values())
    # XXX not sure it is so simple
    return ps.flatten(ps.lift_many(values))


def _what_if_player_chooses_get_mixed(
    ps: PossibilityStructure[Pr],
    choices: Mapping[PlayerName, FrozenSet[Poss[U, Pr]]],
    pure_outcomes: Mapping[JointPureActions, Mapping[PlayerName, UncertainCombined]],
    player_name: PlayerName,
) -> Mapping[JointMixedActions, UncertainCombined]:
    players_ordered = list(choices)
    players_strategies = [choices[_] for _ in players_ordered]
    results: Dict[JointMixedActions, UncertainCombined] = {}
    for choices in itertools.product(*tuple(players_strategies)):
        choice: JointMixedActions = frozendict(zip(players_ordered, choices))

        dist: Poss[JointPureActions, Pr]
        dist = get_mixed2(ps, choice)

        def get_for_me(x: JointPureActions) -> UncertainCombined:
            r = pure_outcomes[x][player_name]

            return r

        mixed_outcome: Poss[UncertainCombined, Pr] = ps.build(dist, get_for_me)
        # TODO: for probabilities, there is something more complicated than just "build"
        # ...
        # logger.info(mixed_outcome=mixed_outcome)
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

    def f(y: JointPureActions) -> JointPureActions:
        return y

    dist: Poss[JointPureActions, Pr] = ps.build_multiple(a=mixed, f=f)

    for _ in dist.support():
        check_joint_pure_actions(_)
    return dist
