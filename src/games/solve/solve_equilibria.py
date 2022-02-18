from typing import Dict, Mapping

from frozendict import frozendict

from dg_commons import fd, PlayerName, RJ, RP, U, X, Y
from games.game_def import Combined, JointMixedActions, JointPureActions, SR, UncertainCombined
from possibilities import Poss
from preferences import Preference
from zuper_commons.types import ZNotImplementedError, ZValueError
from .equilibria import analyze_equilibria, EquilibriaAnalysis
from .solution_security import get_mixed_joint_actions, get_security_policies
from .solution_structures import BAIL_MNE, GameNode, MIX_MNE, SECURITY_MNE, SolvingContext, ValueAndActions
from .. import GameConstants
from ..checks import check_joint_mixed_actions, check_joint_pure_actions

__all__ = ["solve_equilibria", "solve_final_for_everyone"]


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
    preferences: Mapping[PlayerName, Preference[UncertainCombined]]
    preferences = fd({k: sc.outcome_preferences[k] for k in players_active})

    ea: EquilibriaAnalysis[X, U, Y, RP, RJ]
    ea = analyze_equilibria(
        ps=sc.game.ps, gn=gn, solved=solved, preferences=preferences, solver_params=sc.solver_params
    )
    # logger.info(ea=ea)
    n_nondom_nash_equilibria = len(ea.nondom_nash_equilibria)
    if n_nondom_nash_equilibria == 1:
        eq = list(ea.nondom_nash_equilibria)[0]
        check_joint_mixed_actions(eq)
        game_value = dict(ea.nondom_nash_equilibria[eq])
        for player_final, final_value in gn.personal_final_reward.items():
            joint_reward_id = sc.game.joint_reward.joint_reward_identity()
            game_value[player_final] = ps.unit(Combined(final_value, joint=joint_reward_id))
        # Get the game values for the players in a final state
        game_value.update(get_final_value_for_endings(sc=sc, gn=gn))
        if GameConstants.checks:
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

            game_value: Dict[PlayerName, UncertainCombined] = {}
            for player_name in players_active:

                def f(jpa: JointPureActions) -> UncertainCombined:
                    return solved[jpa][player_name]

                game_value[player_name] = ps.join(ps.build(dist, f))

            # Get the game values for the players in a final state
            game_value.update(get_final_value_for_endings(sc=sc, gn=gn))
            if GameConstants.checks:
                if set(game_value) != set(gn.states):
                    raise ZValueError("The game values are incomplete.", game_value=game_value, gn=gn)
            return ValueAndActions(game_value=fd(game_value), mixed_actions=fd(profile))
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
            for player_name in players_active:

                def f(jpa: JointPureActions) -> UncertainCombined:
                    return solved[jpa][player_name]

                game_value[player_name] = ps.join(ps.build(dist, f))

            # Get the game values for the players in a final state
            game_value.update(get_final_value_for_endings(sc=sc, gn=gn))
            if GameConstants.checks:
                if set(game_value) != set(gn.states):
                    raise ZValueError("The game values are incomplete.", game_value=game_value, gn=gn)
            return ValueAndActions(game_value=fd(game_value), mixed_actions=security_policies)
        elif mNE_strategy == BAIL_MNE:
            msg = "Multiple Nash Equilibria"
            raise ZNotImplementedError(msg, ea=ea)
        else:
            msg = "Unknown strategy for multiple Nash Equilibria"
            raise ZNotImplementedError(msg, mNE_strategy=mNE_strategy)
    else:
        msg = "Unable to find Nash Equilibria for the node."
        raise ZNotImplementedError(msg, states=gn.states, ea=ea)


def get_final_joint_value(
    sc: SolvingContext[X, U, Y, RP, RJ, SR], gn: GameNode[X, U, Y, RP, RJ, SR]
) -> Dict[PlayerName, UncertainCombined]:
    """
    Return the ending value for the ones jointly terminating here.
    """
    game_value: Dict[PlayerName, UncertainCombined] = {}

    for player_name, joint in gn.joint_final_rewards.items():
        personal = sc.game.players[player_name].personal_reward_structure.personal_reward_identity()
        game_value[player_name] = sc.game.ps.unit(Combined(personal=personal, joint=joint))

    return game_value


def get_final_value_for_endings(
    sc: SolvingContext[X, U, Y, RP, RJ, SR], gn: GameNode[X, U, Y, RP, RJ, SR]
) -> Dict[PlayerName, UncertainCombined]:
    """Get the values for all the players ending in the node"""
    # first the jointly ending
    game_value: Dict[PlayerName, UncertainCombined] = get_final_joint_value(sc, gn)
    # then the personal ones
    for player_name, personal in gn.personal_final_reward.items():
        if player_name in game_value:
            # The joint ending wins over the personal ending, so we skip the ones that also jointly ended
            continue
        joint_id = sc.game.joint_reward.joint_reward_identity()
        game_value[player_name] = sc.game.ps.unit(Combined(personal=personal, joint=joint_id))
    return game_value


def solve_final_for_everyone(
    sc: SolvingContext[X, U, Y, RP, RJ, SR], gn: GameNode[X, U, Y, RP, RJ, SR]
) -> ValueAndActions[U, RP, RJ]:
    """
    Solves a node which is a final node for everyone
    """
    game_value = get_final_value_for_endings(sc, gn)
    actions = frozendict()
    return ValueAndActions(game_value=fd(game_value), mixed_actions=actions)
