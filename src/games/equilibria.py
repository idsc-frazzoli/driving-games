from dataclasses import dataclass
from typing import Collection, Dict, Generic, Mapping, Set, Tuple

from frozendict import frozendict
from zuper_commons.types import check_isinstance, ZAssertionError, ZValueError

from preferences import (
    COMP_OUTCOMES,
    ComparisonOutcome,
    FIRST_PREFERRED,
    Preference,
    remove_dominated,
    StrictProductPreference,
)
from .comb_utils import get_all_choices_by_players, get_all_combinations
from .game_def import ASet, JointMixedActions, JointPureActions, PlayerName, RJ, RP, U, X, Y
from .structures_solution import check_joint_pure_actions, check_set_outcomes, SetOfOutcomes


# Choice = TypeVar("Choice")
# Choice = ASet[U]
# Mapping[PlayerName, Choice] -> JointMixedActions
# O -> SetOfOutcomes


@dataclass
class PointStats(Generic[X, U, Y, RP, RJ]):
    happy: ASet[PlayerName]
    unhappy: ASet[PlayerName]
    outcome: SetOfOutcomes
    alternatives: Mapping[PlayerName, ASet[ComparisonOutcome]]

    def __post_init__(self):
        check_set_outcomes(self.outcome)


@dataclass
class EquilibriaAnalysis(Generic[X, U, Y, RP, RJ]):
    nondom_nash_equilibria: Mapping[JointPureActions, SetOfOutcomes]
    nash_equilibria: Mapping[JointPureActions, SetOfOutcomes]
    ps: Dict[JointPureActions, PointStats]

    def __post_init__(self):
        for _ in self.ps:
            check_joint_pure_actions(_)
        for _ in self.nondom_nash_equilibria:
            check_joint_pure_actions(_)
        for _ in self.nash_equilibria:
            check_joint_pure_actions(_)


@dataclass
class Combos(Generic[X, U, Y, RP, RJ]):
    all_comb: ASet[JointPureActions]
    player2choices: JointMixedActions

    def __post_init__(self):
        check_isinstance(self.all_comb, frozenset)
        for _ in self.all_comb:
            check_joint_pure_actions(_)


def check_contains_all_combo(
    possibilities: Collection[JointPureActions],
) -> Combos[X, U, Y, RP, RJ]:
    for _ in possibilities:
        check_joint_pure_actions(_)
    # player2choices: Dict[PlayerName, Set[X]] = defaultdict(set)[
    # for actions in action2outcome:
    #     for player_name, action in actions.items():
    #         player2choices[player_name].add(action)
    #         ]
    mixed_actions: JointMixedActions = get_all_choices_by_players(possibilities)

    all_comb: ASet[JointPureActions] = get_all_combinations(mixed_actions=mixed_actions)
    c: JointPureActions
    for c in all_comb:
        check_joint_pure_actions(c)
        if False:  # XXX: bug
            if c not in possibilities:  # pragma: no cover
                msg = "Missing combination"
                raise ZValueError(
                    msg,
                    c=c,
                    id_c=id(c),
                    p=possibilities,
                    type_p=type(possibilities),
                    type_c=type(c),
                    repr_c=repr(c),
                    repr_p=repr(possibilities),
                    id_ps=set(id(_) for _ in possibilities),
                    c_in_p=c in possibilities,
                    c_in_list_p=c in list(possibilities),
                    c_in_fset_p=c in frozenset(possibilities),
                    c_in_set_p=c in set(possibilities),
                    p_contains_c=possibilities.__contains__(c),
                    p_eq_frozen_c=possibilities == frozenset({c}),
                    same_as_first=list(possibilities)[0] == c,
                )
    return Combos(all_comb, mixed_actions)


def analyze_equilibria(
    solved: Mapping[JointPureActions, SetOfOutcomes],
    preferences: Mapping[PlayerName, Preference[SetOfOutcomes]],
) -> EquilibriaAnalysis:
    # we want to make sure that there are all combinations
    combos: Combos[X, U, Y, RP, RJ] = check_contains_all_combo(frozenset(solved))
    player_names = set(combos.player2choices)
    if set(preferences) != set(player_names):  # pragma: no cover
        raise ZValueError(solved=solved, preferences=preferences)
    # logger.info(combos=combos)
    comb: JointPureActions
    ps: Dict[JointPureActions, PointStats] = {}
    x0: JointPureActions
    x1: JointPureActions

    nash_equilibria = {}
    action_to_change: ASet[U]
    for x0 in combos.all_comb:
        if not (x0 in solved):
            raise ZValueError(x0=x0, solved=set(solved))

        check_joint_pure_actions(x0)
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
                zassert(x1 in solved, x1=x1, solved=set(solved))
                o1, o0 = solved[x1], solved[x0]
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
            outcome=solved[x0],
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
        nondom_nash_equilibria=nondom_nash_equilibria, nash_equilibria=nash_equilibria, ps=ps
    )


def zassert(val: bool, **kwargs):
    if not val:  # pragma: no cover
        msg = "Assertion failed"
        raise ZAssertionError(msg, val=val, **kwargs)


def variations(
    c: Combos[X, U, Y, RP, RJ], x0: JointPureActions, player_name: PlayerName
) -> Mapping[U, JointPureActions]:
    check_joint_pure_actions(x0)
    all_actions: Set[U] = set(c.player2choices[player_name])
    current_action: U = x0[player_name]
    assert current_action in all_actions, (current_action, all_actions)
    all_actions.remove(current_action)

    # assert len(all_actions) >= 1, c.player2choices[player_name]
    res = {}
    for alternative in all_actions:
        d = dict(x0)
        d[player_name] = alternative
        _ = frozendict(d)
        check_joint_pure_actions(_)
        res[alternative] = _
    return frozendict(res)
