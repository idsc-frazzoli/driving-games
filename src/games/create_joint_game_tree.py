from collections import defaultdict
from dataclasses import replace
from typing import Dict

from frozendict import frozendict

from .game_def import (
    JointPureActions,
    JointState,
    RJ,
    RP,
    U,
    X,
    Y,
)
from .structures_solution import GameNode, IterationContext

__all__ = []


def create_game_tree(ic: IterationContext, N: JointState) -> GameNode[Pr, X, U, Y, RP, RJ]:
    if N in ic.cache:
        return ic.cache[N]
    states = {k: v for k, v in N.items() if v is not None}
    # if ic.depth > 20:
    #     return None
    # get all possible successors
    G = ic.gp.game_graph

    N2: JointState

    moves = defaultdict(set)

    pure_outcomes: Dict[JointPureActions, GameNode[Pr, X, U, Y, RP, RJ]] = {}

    ic2 = replace(ic, depth=ic.depth + 1)
    # noinspection PyArgumentList
    for N_, N2, attrs in G.out_edges(N, data=True):
        joint_action: JointPureActions = attrs["action"]
        # note: can be null
        # check_joint_pure_actions(joint_action)

        for p, m in joint_action.items():
            if m is not None:
                moves[p].add(m)
        pure_action: JointPureActions
        pure_action = frozendict(
            {pname: action for pname, action in joint_action.items() if action is not None}
        )
        pure_outcomes[pure_action] = create_game_tree(ic2, N2)

    is_final = {}
    for player_name, player_state in states.items():
        _ = ic.gp.game.players[player_name]
        if _.personal_reward_structure.is_personal_final_state(player_state):
            f = _.personal_reward_structure.personal_final_reward(player_state)
            is_final[player_name] = f

    incremental = defaultdict(dict)
    for k, its_moves in moves.items():
        for move in its_moves:
            pri = ic.gp.game.players[k].personal_reward_structure.personal_reward_incremental
            inc = pri(states[k], move, ic.gp.solver_params.dt)
            incremental[k][move] = inc

    who_exits = frozenset(ic.gp.game.joint_reward.is_joint_final_state(states))
    joint_final = who_exits
    if joint_final:
        joint_final_rewards = ic.gp.game.joint_reward.joint_reward(states)
    else:
        joint_final_rewards = {}

    moves = {k: frozenset(v) for k, v in moves.items()}

    outcomes = pure_outcomes
    res = GameNode(
        moves=frozendict(moves),
        states=frozendict(states),
        outcomes=frozendict(outcomes),
        incremental=frozendict({k: frozendict(v) for k, v in incremental.items()}),
        joint_final_rewards=frozendict(joint_final_rewards),
        is_final=frozendict(is_final),
    )
    ic.cache[N] = res
    return res
