from typing import Dict, Mapping, MutableMapping

from frozendict import frozendict

from games.solve.solution_security import get_mixed_joint_actions, get_security_policies
from possibilities import Poss
from preferences import Preference
from zuper_commons.types import ZNotImplementedError, ZValueError

from .equilibria import analyze_equilibria, EquilibriaAnalysis
from games import logger
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

    ea = analyze_equilibria(
        ps=sc.game.ps, gn=gn, solved=solved, preferences=preferences, solver_params=sc.solver_params
    )

    # toc = perf_counter() - tic
    # logger.info(f"Time taken to analyze equilibria: {toc:.2f} [s]")
    # logger.info(ea=ea)
    if len(ea.nondom_nash_equilibria) == 1:

        eq = list(ea.nondom_nash_equilibria)[0]
        check_joint_mixed_actions(eq)

        game_value = dict(ea.nondom_nash_equilibria[eq])

        # Get the game values for the players in a final state
        game_value.update(
                get_game_values_final(sc=sc, gn=gn)
            )

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
            for player_name in players_active:  # check only active players

                def f(jpa: JointPureActions) -> UncertainCombined:
                    return solved[jpa][player_name]

                game_value1[player_name] = ps.join(ps.build(dist, f))

            # logger.info(dist=dist)
            # game_value1 = ps.join(ps.build(dist, solved.__getitem__))

            # Get the game values for the players in a final state
            game_value1.update(
                get_game_values_final(sc=sc, gn=gn)
            )

            if set(game_value1) != set(gn.states):
                raise ZValueError("incomplete", game_value=game_value1, gn=gn)

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
            for player_name in players_active:  # check only active players

                def f(jpa: JointPureActions) -> UncertainCombined:
                    return solved[jpa][player_name]

                game_value[player_name] = ps.join(ps.build(dist, f))

            # Get the game values for the players in a final state
            game_value.update(
                get_game_values_final(sc=sc, gn=gn)
            )

            if set(game_value) != set(gn.states):
                raise ZValueError("incomplete", game_value=game_value, gn=gn)

            # game_value: Mapping[PlayerName, UncertainCombined]
            game_value_ = fd(game_value)
            return ValueAndActions(game_value=game_value_, mixed_actions=security_policies)
        elif mNE_strategy == BAIL_MNE:
            msg = "Multiple Nash Equilibria"
            raise ZNotImplementedError(msg, ea=ea)
        else:
            assert False, mNE_strategy


def get_game_values_final(sc: SolvingContext[X, U, Y, RP, RJ, SR], gn: GameNode[X, U, Y, RP, RJ, SR]) -> Mapping[PlayerName, UncertainCombined]:
    """
    Get the game value of all the final states
    """

    pl_final = set(gn.is_final)  # Set of players that are at personal final state
    pl_joint_final = set(gn.joint_final_rewards)  # set of players that are jointly final
    pl_joint_and_personal = pl_final & pl_joint_final  # get the players who are both joint and personal final

    game_value: Dict[PlayerName, UncertainCombined] = {}


    # iterate through all the players that finished and extract the their reward
    for player_name in (pl_final | pl_joint_final):

        if player_name in pl_joint_and_personal:
            # The player is both joint and personal final, extract both cost
            personal = gn.is_final[player_name]
            joint = gn.joint_final_rewards[player_name]
            game_value[player_name] = sc.game.ps.unit(Combined(personal=personal, joint=joint))
        else:
            if player_name in pl_final:
                # The player is personal final only, extract only the personal coste
                personal = gn.is_final[player_name]
                game_value[player_name] = sc.game.ps.unit(Combined(personal=personal, joint=None))
            elif player_name in pl_joint_final:
                # player is joint final, extract the joint final cost and get the identity cost as personal cost
                personal = sc.game.players[player_name].personal_reward_structure.personal_reward_identity()
                joint = gn.joint_final_rewards[player_name]
                game_value[player_name] = sc.game.ps.unit(Combined(personal=personal, joint=joint))
            else:
                assert False, "Should not happen"

    game_value_ = frozendict(game_value)
    return game_value_


