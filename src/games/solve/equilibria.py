import itertools
from dataclasses import dataclass
from typing import Dict, FrozenSet, Generic, Mapping, Set

from frozendict import frozendict

from possibilities import Poss, PossibilityMonad
from preferences import (
    COMP_OUTCOMES,
    ComparisonOutcome,
    FIRST_PREFERRED,
    Preference,
    remove_dominated,
    StrictProductPreferenceDict,
)
from zuper_commons.types import ZValueError
from games import GameConstants
from games.game_def import (
    check_joint_mixed_actions,
    JointMixedActions,
    JointPureActions,
    PlayerName,
    RJ,
    RP,
    SR,
    U,
    UncertainCombined,
    X,
    Y,
)
from .solution_structures import GameNode
from games.utils import valmap

__all__ = []


@dataclass
class PointStats(Generic[X, U, Y, RP, RJ]):
    happy: FrozenSet[PlayerName]
    unhappy: FrozenSet[PlayerName]
    outcome: Mapping[PlayerName, UncertainCombined]
    alternatives: Mapping[PlayerName, FrozenSet[ComparisonOutcome]]

    def __post_init__(self) -> None:
        if not GameConstants.checks:
            return


@dataclass
class EquilibriaAnalysis(Generic[X, U, Y, RP, RJ]):
    player_mixed_strategies: Mapping[PlayerName, FrozenSet[Poss[U]]]
    nondom_nash_equilibria: Mapping[JointMixedActions, Mapping[PlayerName, UncertainCombined]]
    nash_equilibria: Mapping[JointMixedActions, Mapping[PlayerName, UncertainCombined]]
    ps: Dict[JointMixedActions, PointStats]

    def __post_init__(self) -> None:
        if not GameConstants.checks:
            return

        for _ in self.ps:
            check_joint_mixed_actions(_)
        for _ in self.nondom_nash_equilibria:
            check_joint_mixed_actions(_)
        for _ in self.nash_equilibria:
            check_joint_mixed_actions(_)


def analyze_equilibria(
    *,
    ps: PossibilityMonad,
    gn: GameNode[X, U, Y, RP, RJ, SR],
    solved: Mapping[JointPureActions, Mapping[PlayerName, UncertainCombined]],
    preferences: Mapping[PlayerName, Preference[UncertainCombined]],
) -> EquilibriaAnalysis:
    # Now we want to find all mixed strategies
    # Example: From sets, you could have [A, B] ->  {A}, {B}, {A,B}
    # Example: From probs, you could have [A,B] -> {A:1}, {B:1} , {A:0.5, B:0.5}, ...
    # todo for probabilities this is restrictive...(mix returns a finite set)
    player_mixed_strategies: Dict[PlayerName, FrozenSet[Poss[U]]] = valmap(ps.mix, gn.moves)
    # logger.info(player_mixed_strategies=player_mixed_strategies)
    # now we do the product of the mixed strategies
    # let's order them

    players_ordered = list(player_mixed_strategies)  # only the active ones
    players_strategies = [player_mixed_strategies[_] for _ in players_ordered]
    all_players = set(gn.states)
    active_players = set(gn.moves)

    results: Dict[JointMixedActions, Mapping[PlayerName, UncertainCombined]] = {}
    for choices in itertools.product(*tuple(players_strategies)):
        choice: JointMixedActions = frozendict(zip(players_ordered, choices))

        def f(y: JointPureActions) -> JointPureActions:
            return y

        dist: Poss[JointPureActions] = ps.build_multiple(a=choice, f=f)

        mixed_outcome: Poss[Mapping[PlayerName, UncertainCombined]]
        mixed_outcome = ps.build(dist, solved.__getitem__)
        res: Dict[PlayerName, UncertainCombined] = {}
        for player_name in active_players:  # all of them, not only the active ones

            def g(_: Mapping[PlayerName, UncertainCombined]) -> UncertainCombined:
                if player_name not in _:
                    msg = f"Cannot get value for {player_name!r}."
                    raise ZValueError(
                        msg,
                        player_name=player_name,
                        _=_,
                        mixed_outcome=mixed_outcome,
                        solved=solved,
                        gn=gn,
                    )
                return _[player_name]

            x = ps.join(ps.build(mixed_outcome, g))
            res[player_name] = x

        results[choice] = frozendict(res)
        # results[choice] = solved[choice]
    # logger.info(results=results)
    return analyze(player_mixed_strategies, results, preferences)


def analyze(
    player_mixed_strategies: Mapping[PlayerName, FrozenSet[Poss[U]]],
    results: Mapping[JointMixedActions, Mapping[PlayerName, UncertainCombined]],
    preferences: Mapping[PlayerName, Preference[UncertainCombined]],
):
    # logger.info(combos=combos)
    comb: JointPureActions
    ps: Dict[JointPureActions, PointStats] = {}
    a0: JointMixedActions
    a1: JointMixedActions
    player_names = set(player_mixed_strategies)
    nash_equilibria = {}
    action_to_change: FrozenSet[U]
    for a0 in results:
        happy_players = set()
        unhappy_players = set()
        alternatives = {}
        for player_name in player_names:
            pref: Preference[UncertainCombined]
            try:
                pref = preferences[player_name]
            except:
                pref = preferences[player_name[0]]
            is_happy: bool = True
            variations_: Mapping[U, JointMixedActions]
            variations_ = variations(player_mixed_strategies, a0, player_name)
            alternatives_player = {}
            # logger.info('looking for variations', variations_=variations_)
            for action_to_change, a1 in variations_.items():
                # zassert(x1 in results, a1=a1, results=set(results))
                o0: UncertainCombined
                o1: UncertainCombined
                try:
                    o1, o0 = results[a1][player_name], results[a0][player_name]
                except:
                    o1, o0 = results[a1][player_name[0]], results[a0][player_name[0]]
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
            outcome=results[a0],
            alternatives=frozendict(alternatives),
        )
        ps[a0] = stats

        if not unhappy_players:
            nash_equilibria[a0] = stats.outcome
    # logger.info(ps=ps)

    # compare product of monadic outcomes
    pref: Preference[Mapping[PlayerName, UncertainCombined]] = StrictProductPreferenceDict(preferences)

    # logger.info(nash_equilibria=nash_equilibria, preferences=preferences, pref=pref)
    nondom_nash_equilibria = remove_dominated(nash_equilibria, pref)

    return EquilibriaAnalysis(
        player_mixed_strategies=player_mixed_strategies,
        nondom_nash_equilibria=nondom_nash_equilibria,
        nash_equilibria=nash_equilibria,
        ps=ps,
    )


#
# def zassert(val: bool, **kwargs):
#     if not val:  # pragma: no cover
#         msg = "Assertion failed"
#         raise ZAssertionError(msg, val=val, **kwargs)
#


def variations(
    player_mixed_strategies: Mapping[PlayerName, FrozenSet[Poss[U]]],
    x0: Mapping[PlayerName, Poss[U]],
    player_name: PlayerName,
) -> Mapping[U, Mapping[PlayerName, Poss[U]]]:
    # check_joint_pure_actions(x0)
    all_mixed_actions: Set[Poss[U]] = set(player_mixed_strategies[player_name])
    current_action: Poss[U] = x0[player_name]
    assert current_action in all_mixed_actions, (current_action, all_mixed_actions)
    all_mixed_actions.remove(current_action)

    # assert len(all_actions) >= 1, c.player2choices[player_name]
    res = {}
    for alternative in all_mixed_actions:
        d = dict(x0)
        d[player_name] = alternative
        _ = frozendict(d)

        res[alternative] = _
    return frozendict(res)
