from typing import Dict, Mapping, MutableMapping

from frozendict import frozendict

from games.solve.solution_security import get_mixed_joint_actions, get_security_policies
from possibilities import Poss
from preferences import Preference
from zuper_commons.types import ZNotImplementedError, ZValueError

from .equilibria import analyze_equilibria, EquilibriaAnalysis
from games.game_def import (
    check_joint_mixed_actions,
    check_joint_pure_actions,
    Combined,
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
from .solution_structures import (
    GameNode,
    SolvingContext,
    BAIL_MNE,
    MIX_MNE,
    SECURITY_MNE,
    ValueAndActions,
)
from games.utils import fd


def solve_equilibria(
    sc: SolvingContext[X, U, Y, RP, RJ, SR],
    gn: GameNode[X, U, Y, RP, RJ, SR],
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
    # tic = perf_counter()
    ea = analyze_equilibria(ps=sc.game.ps, gn=gn, solved=solved, preferences=preferences)
    # toc = perf_counter() - tic
    # logger.info(f"Time taken to analyze equilibria: {toc:.2f} [s]")
    # logger.info(ea=ea)
    if len(ea.nondom_nash_equilibria) == 1:

        eq = list(ea.nondom_nash_equilibria)[0]
        check_joint_mixed_actions(eq)

        game_value = dict(ea.nondom_nash_equilibria[eq])
        for player_final, final_value in gn.is_final.items():
            game_value[player_final] = ps.unit(Combined(final_value, None))
        if set(game_value) != set(gn.states):
            raise ZValueError("incomplete", game_value=game_value, gn=gn)
        return ValueAndActions(game_value=frozendict(game_value), mixed_actions=eq)
    else:
        # multiple non-dominated nash equilibria
        outcomes = set(ea.nondom_nash_equilibria.values())

        mNE_strategy = sc.solver_params.strategy_multiple_nash
        if mNE_strategy == MIX_MNE:
            # fixme: Not really sure this makes sense when there are probabilities

            profile: Dict[PlayerName, Poss[U]] = {}
            for player_name in players_active:
                # find all the mixed strategies he would play at equilibria
                res = set()
                for joint_mixed_actions in ea.nondom_nash_equilibria:
                    res.add(joint_mixed_actions[player_name])
                mNE_strategy = ps.join(ps.lift_many(res))
                # check_poss(mNE_strategy)
                profile[player_name] = mNE_strategy

            def f(y: JointPureActions) -> JointPureActions:
                return frozendict(y)

            dist: Poss[JointPureActions] = ps.build_multiple(a=profile, f=f)

            game_value1: MutableMapping[PlayerName, UncertainCombined]
            game_value1 = {}
            for player_name in gn.states:

                def f(jpa: JointPureActions) -> UncertainCombined:
                    return solved[jpa][player_name]

                game_value1[player_name] = ps.join(ps.build(dist, f))

            # logger.info(dist=dist)
            # game_value1 = ps.join(ps.build(dist, solved.__getitem__))

            return ValueAndActions(game_value=fd(game_value1), mixed_actions=fd(profile))
        # Anything can happen
        elif mNE_strategy == SECURITY_MNE:
            # fixme: Not really sure this makes sense when there are probabilities
            # fixme for probabilities
            security_policies: JointMixedActions
            security_policies = get_security_policies(ps, solved, sc.outcome_preferences, ea)
            check_joint_mixed_actions(security_policies)
            dist: Poss[JointPureActions]
            dist = get_mixed_joint_actions(ps, security_policies)
            # logger.info(dist=dist)
            for joint_mixed_actions in dist.support():
                check_joint_pure_actions(joint_mixed_actions)
            # logger.info(dist=dist)
            game_value = {}
            for player_name in gn.states:

                def f(jpa: JointPureActions) -> UncertainCombined:
                    return solved[jpa][player_name]

                game_value[player_name] = ps.join(ps.build(dist, f))

            # game_value: Mapping[PlayerName, UncertainCombined]
            game_value_ = fd(game_value)
            return ValueAndActions(game_value=game_value_, mixed_actions=security_policies)
        elif mNE_strategy == BAIL_MNE:
            msg = "Multiple Nash Equilibria"
            raise ZNotImplementedError(msg, ea=ea)
        else:
            assert False, mNE_strategy
