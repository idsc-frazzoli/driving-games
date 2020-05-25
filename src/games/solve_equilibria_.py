from typing import Dict, Mapping

from frozendict import frozendict

from games.solution_security import get_mixed2, get_security_policies
from possibilities import Poss
from preferences import Preference
from zuper_commons.types import ZNotImplementedError, ZValueError
from . import logger
from .equilibria import analyze_equilibria, EquilibriaAnalysis
from .game_def import (
    check_joint_mixed_actions2,
    check_joint_pure_actions,
    Combined,
    JointMixedActions,
    JointPureActions,
    PlayerName,
    Pr,
    RJ,
    RP,
    SR,
    U,
    UncertainCombined,
    X,
    Y,
)
from .structures_solution import (
    GameNode,
    SolvingContext,
    STRATEGY_BAIL,
    STRATEGY_MIX,
    STRATEGY_SECURITY,
    ValueAndActions2,
)
from .utils import fd


def solve_equilibria(
    sc: SolvingContext[Pr, X, U, Y, RP, RJ, SR],
    gn: GameNode[Pr, X, U, Y, RP, RJ, SR],
    solved: Mapping[JointPureActions, Mapping[PlayerName, UncertainCombined]],
) -> ValueAndActions2[Pr, U, RP, RJ]:
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
    preferences = {k: sc.outcome_set_preferences[k] for k in players_active}

    ea: EquilibriaAnalysis[Pr, X, U, Y, RP, RJ]
    ea = analyze_equilibria(ps=sc.game.ps, gn=gn, solved=solved, preferences=preferences)
    # logger.info(ea=ea)
    if len(ea.nondom_nash_equilibria) == 1:
        eq = list(ea.nondom_nash_equilibria)[0]
        check_joint_mixed_actions2(eq)

        game_value = dict(ea.nondom_nash_equilibria[eq])
        for player_final, final_value in gn.is_final.items():
            game_value[player_final] = ps.lift_one(Combined(final_value, None))
        if set(game_value) != set(gn.states):
            raise ZValueError("incomplete", game_value=game_value, gn=gn)
        return ValueAndActions2(game_value=frozendict(game_value), mixed_actions=eq)
    else:
        # multiple nondominated, but same outcome

        # multiple non-dominated nash equilibria
        outcomes = set(ea.nondom_nash_equilibria.values())

        strategy = sc.solver_params.strategy_multiple_nash
        if strategy == STRATEGY_MIX:
            # XXX: Not really sure this makes sense when there are probabilities
            profile: Dict[PlayerName, Poss[U, Pr]] = {}
            for player_name in players_active:
                # find all the mixed strategies he would play at equilibria
                res = set()
                for _ in ea.nondom_nash_equilibria:
                    res.add(_[player_name])
                strategy = ps.flatten(ps.lift_many(res))
                # check_poss(strategy)
                profile[player_name] = strategy

            def f(y: JointPureActions) -> JointPureActions:
                return frozendict(y)

            dist: Poss[JointPureActions, Pr] = ps.build_multiple(a=profile, f=f)

            game_value1: Mapping[PlayerName, UncertainCombined]
            game_value1 = {}
            for player_name in gn.states:

                def f(jpa: JointPureActions) -> UncertainCombined:
                    return solved[jpa][player_name]

                game_value1[player_name] = ps.flatten(ps.build(dist, f))

            # logger.info(dist=dist)
            # game_value1 = ps.flatten(ps.build(dist, solved.__getitem__))

            return ValueAndActions2(game_value=fd(game_value1), mixed_actions=frozendict(profile))
        # Anything can happen
        elif strategy == STRATEGY_SECURITY:
            security_policies: JointMixedActions
            security_policies = get_security_policies(ps, solved, sc.outcome_set_preferences, ea)
            check_joint_mixed_actions2(security_policies)
            dist: Poss[JointPureActions, Pr]
            dist = get_mixed2(ps, security_policies)
            # logger.info(dist=dist)
            for _ in dist.support():
                check_joint_pure_actions(_)
            # logger.info(dist=dist)
            game_value = {}
            for player_name in gn.states:

                def f(jpa: JointPureActions) -> UncertainCombined:
                    return solved[jpa][player_name]

                game_value[player_name] = ps.flatten(ps.build(dist, f))

            # game_value: Mapping[PlayerName, UncertainCombined]
            # game_value = ps.flatten(ps.build(dist, solved.__getitem__))
            game_value_ = fd(game_value)
            return ValueAndActions2(game_value=game_value_, mixed_actions=security_policies)
        elif strategy == STRATEGY_BAIL:
            msg = "Multiple Nash Equilibria"
            raise ZNotImplementedError(msg, ea=ea)
        else:
            assert False, strategy
