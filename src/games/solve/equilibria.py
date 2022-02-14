import itertools
from dataclasses import dataclass
from typing import Dict, FrozenSet, Generic, Mapping, Set

from zuper_commons.types import ZValueError, ZNotImplementedError

from dg_commons import valmap, PlayerName, RJ, RP, U, X, Y, fd
from games import GameConstants
from games.checks import check_joint_mixed_actions
from games.game_def import JointMixedActions, JointPureActions, SR, UncertainCombined, PlayerOptions
from possibilities import Poss, PossibilityMonad, PossibilitySet
from preferences import (
    COMP_OUTCOMES,
    ComparisonOutcome,
    FIRST_PREFERRED,
    Preference,
    remove_dominated,
    StrictProductPreferenceDict,
)
from .solution_structures import (
    GameNode,
    SolverParams,
    FINITE_MIX_STRATEGIES,
    MIX_STRATEGIES,
    PURE_STRATEGIES,
)

__all__ = ["EquilibriaAnalysis"]


@dataclass
class PointStats(Generic[X, U, Y, RP, RJ]):
    happy: FrozenSet[PlayerName]
    unhappy: FrozenSet[PlayerName]
    outcome: Mapping[PlayerName, UncertainCombined]
    alternatives: Mapping[PlayerName, FrozenSet[ComparisonOutcome]]

    def __post_init__(self):
        if not GameConstants.checks:
            return


@dataclass
class EquilibriaAnalysis(Generic[X, U, Y, RP, RJ]):
    player_mixed_strategies: Mapping[PlayerName, FrozenSet[Poss[U]]]
    """The different strategy profiles for each player"""
    nondom_nash_equilibria: Mapping[JointMixedActions, Mapping[PlayerName, UncertainCombined]]
    """The non-dominated NE strategy profiles and the corresponding outcome."""
    nash_equilibria: Mapping[JointMixedActions, Mapping[PlayerName, UncertainCombined]]
    """The NE strategy profiles and the corresponding outcome."""
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


def _get_admissible_strategies(
    ps: PossibilityMonad, moves: PlayerOptions, solver_params: SolverParams
) -> Dict[PlayerName, FrozenSet[Poss[U]]]:
    """
    Now we want to find all mixed strategies
    Example: From sets, you could have [A, B] ->  {A}, {B}, {A,B}
    Example: From probs, you could have [A,B] -> {A:1}, {B:1} , {A:0.5, B:0.5}, ...
    Note that pure strategies are considered singleton mixed strategies
    # todo implement mixed strategies with ProbabilityMonad
    :param ps:
    :param moves:
    :param solver_params:
    :return:
    """
    adms_strategies = solver_params.admissible_strategies
    is_set_monad = isinstance(ps, PossibilitySet)
    if adms_strategies == PURE_STRATEGIES:

        def _f(player_options):
            return frozenset(map(ps.unit, player_options))

        return valmap(_f, moves)
    elif adms_strategies == FINITE_MIX_STRATEGIES or (adms_strategies == MIX_STRATEGIES and is_set_monad):
        return valmap(ps.mix, moves)
    elif adms_strategies == MIX_STRATEGIES:
        raise ZNotImplementedError("Mix strategies are not implemented either than for the SetMonad")
    else:
        raise ZValueError("not recognized value")


def analyze_equilibria(
    *,
    ps: PossibilityMonad,
    gn: GameNode[X, U, Y, RP, RJ, SR],
    solved: Mapping[JointPureActions, Mapping[PlayerName, UncertainCombined]],
    preferences: Mapping[PlayerName, Preference[UncertainCombined]],
    solver_params: SolverParams,
) -> EquilibriaAnalysis:
    player_mixed_strategies: Dict[PlayerName, FrozenSet[Poss[U]]]
    player_mixed_strategies = _get_admissible_strategies(ps=ps, moves=gn.moves, solver_params=solver_params)

    # now we do the product of the mixed strategies, let's order them
    players_ordered = list(player_mixed_strategies)  # only the active ones
    players_strategies = [player_mixed_strategies[_] for _ in players_ordered]
    all_players = set(gn.states)
    active_players = set(gn.moves)

    results: Dict[JointMixedActions, Mapping[PlayerName, UncertainCombined]] = {}
    for choices in itertools.product(*tuple(players_strategies)):
        choice: JointMixedActions = fd(zip(players_ordered, choices))

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

        results[choice] = fd(res)
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
            pref: Preference[UncertainCombined] = preferences[player_name]
            is_happy: bool = True
            variations_: Mapping[U, JointMixedActions]
            variations_ = variations(player_mixed_strategies, a0, player_name)
            alternatives_player = {}
            # logger.info('looking for variations', variations_=variations_)
            for action_to_change, a1 in variations_.items():
                # zassert(x1 in results, a1=a1, results=set(results))
                o0: UncertainCombined = results[a0][player_name]
                o1: UncertainCombined = results[a1][player_name]
                res = pref.compare(o1, o0)
                assert res in COMP_OUTCOMES, (res, pref)
                # logger.info(o1=o1, o0=o0, res=res)
                if res == FIRST_PREFERRED:
                    is_happy = False
                alternatives_player[action_to_change] = res
            alternatives[player_name] = fd(alternatives_player)
            if is_happy:
                happy_players.add(player_name)
            else:
                unhappy_players.add(player_name)
        stats = PointStats(
            happy=frozenset(happy_players),
            unhappy=frozenset(unhappy_players),
            outcome=results[a0],
            alternatives=fd(alternatives),
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
        _ = fd(d)

        res[alternative] = _
    return fd(res)
