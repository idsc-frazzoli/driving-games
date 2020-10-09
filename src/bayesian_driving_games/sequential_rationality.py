import itertools
from typing import Mapping, Dict, FrozenSet

from frozendict import frozendict
from zuper_commons.types import ZValueError, ZNotImplementedError

from bayesian_driving_games.structures_solution import BayesianSolvingContext, BayesianGameNode
from games import JointPureActions, PlayerName
from games.equilibria import EquilibriaAnalysis, analyze
from games.game_def import (
    UncertainCombined,
    U,
    RP,
    RJ,
    check_joint_pure_actions,
    X,
    Y,
    check_joint_mixed_actions2,
    Combined,
    JointMixedActions,
)
from games.solution_security import get_security_policies, get_mixed2
from games.structures_solution import ValueAndActions, STRATEGY_MIX, STRATEGY_SECURITY, STRATEGY_BAIL
from games.utils import fd, valmap
from possibilities import Poss, PossibilityMonad
from preferences import Preference


def solve_sequential_rationality(
    sc: BayesianSolvingContext,
    gn: BayesianGameNode,
    solved: Mapping[JointPureActions, Mapping[PlayerName, UncertainCombined]],
) -> ValueAndActions[U, RP, RJ]:
    ps = sc.game.ps
    for pure_action in solved:
        check_joint_pure_actions(pure_action)

    if not gn.moves:
        msg = "Cannot solve_equilibria if there are no moves "
        raise ZValueError(msg, gn=gn)
    # logger.info(gn=gn, solved=solved)
    # logger.info(possibilities=list(solved))
    players_active = set(gn.moves)
    preferences: Dict[PlayerName, Preference[UncertainCombined]]
    preferences = {k: sc.outcome_preferences[k] for k in players_active}

    ea: EquilibriaAnalysis[X, U, Y, RP, RJ]
    ea = analyze_sequential_rational(ps=sc.game.ps, gn=gn, solved=solved, preferences=preferences)
    # logger.info(ea=ea)
    if len(ea.nondom_nash_equilibria) == 1:

        eq = list(ea.nondom_nash_equilibria)[0]
        check_joint_mixed_actions2(eq)

        game_value = dict(ea.nondom_nash_equilibria[eq])
        for player_final, final_value in gn.is_final.items():
            game_value[player_final] = ps.unit(Combined(final_value, None))
        if set(game_value) != set(gn.states):
            raise ZValueError("incomplete", game_value=game_value, gn=gn)
        return ValueAndActions(game_value=frozendict(game_value), mixed_actions=eq)
    else:
        # multiple non-dominated nash equilibria
        outcomes = set(ea.nondom_nash_equilibria.values())

        strategy = sc.solver_params.strategy_multiple_nash
        if strategy == STRATEGY_MIX:
            # fixme: Not really sure this makes sense when there are probabilities
            profile: Dict[PlayerName, Poss[U]] = {}
            for player_name in players_active:
                # find all the mixed strategies he would play at equilibria
                res = set()
                for _ in ea.nondom_nash_equilibria:
                    res.add(_[player_name])
                strategy = ps.join(ps.lift_many(res))
                # check_poss(strategy)
                profile[player_name] = strategy

            def f(y: JointPureActions) -> JointPureActions:
                return frozendict(y)

            dist: Poss[JointPureActions] = ps.build_multiple(a=profile, f=f)

            game_value1: Mapping[PlayerName, UncertainCombined]
            game_value1 = {}
            for player_name in gn.states:

                def f(jpa: JointPureActions) -> UncertainCombined:
                    return solved[jpa][player_name]

                game_value1[player_name] = ps.join(ps.build(dist, f))

            # logger.info(dist=dist)
            # game_value1 = ps.join(ps.build(dist, solved.__getitem__))

            return ValueAndActions(game_value=fd(game_value1), mixed_actions=frozendict(profile))
        # Anything can happen
        elif strategy == STRATEGY_SECURITY:
            security_policies: JointMixedActions
            security_policies = get_security_policies(ps, solved, sc.outcome_preferences, ea)
            check_joint_mixed_actions2(security_policies)
            dist: Poss[JointPureActions]
            dist = get_mixed2(ps, security_policies)
            # logger.info(dist=dist)
            for _ in dist.support():
                check_joint_pure_actions(_)
            # logger.info(dist=dist)
            game_value = {}
            for player_name in gn.states:

                def f(jpa: JointPureActions) -> UncertainCombined:
                    return solved[jpa][player_name]

                game_value[player_name] = ps.join(ps.build(dist, f))

            # game_value: Mapping[PlayerName, UncertainCombined]
            game_value_ = fd(game_value)
            return ValueAndActions(game_value=game_value_, mixed_actions=security_policies)
        elif strategy == STRATEGY_BAIL:
            msg = "Multiple Nash Equilibria"
            raise ZNotImplementedError(msg, ea=ea)
        else:
            assert False, strategy


def analyze_sequential_rational(
    *,
    ps: PossibilityMonad,
    gn: BayesianGameNode,
    solved: Mapping[JointPureActions, Mapping[PlayerName, UncertainCombined]],
    preferences: Mapping[PlayerName, Preference[UncertainCombined]],
) -> EquilibriaAnalysis:
    # Now we want to find all mixed strategies
    # Example: From sets, you could have [A, B] ->  {A}, {B}, {A,B}
    # Example: From probs, you could have [A,B] -> {A:1}, {B:1} , {A:0.5, B:0.5}, ...
    # fixme for probabilities this is restrictive... double check mix method
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
                        msg, player_name=player_name, _=_, mixed_outcome=mixed_outcome, solved=solved, gn=gn,
                    )
                return _[player_name]

            x = ps.join(ps.build(mixed_outcome, g))
            res[player_name] = x

        results[choice] = frozendict(res)
        # results[choice] = solved[choice]
    # logger.info(results=results)
    return analyze(player_mixed_strategies, results, preferences)
