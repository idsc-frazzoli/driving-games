from typing import Dict, Mapping, MutableMapping

from frozendict import frozendict
from zuper_commons.types import ZNotImplementedError, ZValueError

from dg_commons import fd, X, U, Y, RP, RJ, PlayerName
from games.game_def import (
    JointMixedActions,
    JointPureActions,
    SR,
    Combined,
    UncertainCombined,
)
from games.solve.equilibria import analyze_equilibria, EquilibriaAnalysis
from games.solve.solution_security import get_mixed_joint_actions, get_security_policies
from possibilities import Poss
from preferences import Preference
from .solution_structures import (
    GameNode,
    SolvingContext,
    BAIL_MNE,
    MIX_MNE,
    SECURITY_MNE,
    ValueAndActions,
)
from .. import GameConstants
from ..checks import check_joint_mixed_actions, check_joint_pure_actions

__all__ = ["solve_equilibria", "solve_final_personal_both", "solve_final_joint"]


def solve_equilibria(
    sc: SolvingContext[X, U, Y, RP, RJ, SR],
    gn: GameNode[X, U, Y, RP, RJ, SR],
    solved: Mapping[JointPureActions, Mapping[PlayerName, UncertainCombined]],
) -> ValueAndActions[U, RP, RJ]:
    """#todo"""
    ps = sc.game.ps
    if GameConstants.checks:
        for pure_action in solved:
            check_joint_pure_actions(pure_action)
        if not gn.moves:
            msg = "Cannot solve_equilibria if there are no moves."
            raise ZValueError(msg, gn=gn)

    # logger.info(gn=gn, solved=solved)
    # logger.info(possibilities=list(solved))
    players_active = set(gn.moves)
    preferences: Dict[PlayerName, Preference[UncertainCombined]]
    preferences = {k: sc.outcome_preferences[k] for k in players_active}

    ea: EquilibriaAnalysis[X, U, Y, RP, RJ]
    # tic = perf_counter()
    ea = analyze_equilibria(
        ps=sc.game.ps, gn=gn, solved=solved, preferences=preferences, solver_params=sc.solver_params
    )
    # toc = perf_counter() - tic
    # logger.info(f"Time taken to analyze equilibria: {toc:.2f} [s]")
    # logger.info(ea=ea)
    n_nondom_nash_equilibria = len(ea.nondom_nash_equilibria)
    if n_nondom_nash_equilibria == 1:
        eq = list(ea.nondom_nash_equilibria)[0]
        check_joint_mixed_actions(eq)
        game_value = dict(ea.nondom_nash_equilibria[eq])
        for player_final, final_value in gn.personal_final_reward.items():
            joint_reward_id = sc.game.joint_reward.joint_reward_identity()
            game_value[player_final] = ps.unit(Combined(final_value, joint=joint_reward_id))
        if set(game_value) != set(gn.states):
            raise ZValueError("The game values are incomplete.", game_value=game_value, gn=gn)
        return ValueAndActions(game_value=fd(game_value), mixed_actions=eq)
    elif n_nondom_nash_equilibria > 1:
        # multiple non-dominated nash equilibria
        # outcomes = set(ea.nondom_nash_equilibria.values())
        mNE_strategy = sc.solver_params.strategy_multiple_nash
        if mNE_strategy == MIX_MNE:
            # This might result in off equilibrium policy at game time.
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
                return fd(y)

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
            # fixme: This makes sense when there are probabilities
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
            msg = "Unknown strategy for multiple Nash Equilibria"
            raise ZNotImplementedError(msg, mNE_strategy=mNE_strategy)
    else:
        msg = "Unable to find Nash Equilibria for the node."
        raise ZNotImplementedError(msg, ea=ea)


def solve_final_joint(
    sc: SolvingContext[X, U, Y, RP, RJ, SR], gn: GameNode[X, U, Y, RP, RJ, SR]
) -> ValueAndActions[U, RP, RJ]:
    """
    Solves a node which is a joint final node
    """
    game_value: Dict[PlayerName, UncertainCombined] = {}

    for player_name, joint in gn.joint_final_rewards.items():
        personal = sc.game.players[player_name].personal_reward_structure.personal_reward_identity()
        game_value[player_name] = sc.game.ps.unit(Combined(personal=personal, joint=joint))

    game_value_ = fd(game_value)
    actions = frozendict()
    return ValueAndActions(game_value=game_value_, mixed_actions=actions)


def solve_final_personal_both(
    sc: SolvingContext[X, U, Y, RP, RJ, SR], gn: GameNode[X, U, Y, RP, RJ, SR]
) -> ValueAndActions[U, RP, RJ]:
    """
    Solves end game node which is final for both players (but not jointly final)
    :param sc:
    :param gn:
    :return:
    """
    game_value: Dict[PlayerName, UncertainCombined] = {}
    for player_name, personal in gn.personal_final_reward.items():
        joint_id = sc.game.joint_reward.joint_reward_identity()
        game_value[player_name] = sc.game.ps.unit(Combined(personal=personal, joint=joint_id))
    game_value_ = fd(game_value)
    actions = frozendict()
    return ValueAndActions(game_value=game_value_, mixed_actions=actions)
