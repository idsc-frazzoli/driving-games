from collections import defaultdict
from dataclasses import dataclass, replace
from decimal import Decimal as D
from typing import AbstractSet, Dict, Generic, Mapping, Optional, Tuple

from frozendict import frozendict
from toolz import itemmap

from possibilities import Poss
from . import logger
from .game_def import Game, JointPureActions, JointState, PlayerName, Pr, RJ, RP, SR, U, X, Y
from .structures_solution import GameFactorization, GameGraph, GameNode
from .utils import fkeyfilter, fvalmap, iterate_dict_combinations

__all__ = []


@dataclass
class IterationContext(Generic[Pr, X, U, Y, RP, RJ, SR]):
    game: Game[Pr, X, U, Y, RP, RJ, SR]
    dt: D
    # gp: GamePreprocessed[Pr, X, U, Y, RP, RJ, SR]
    cache: Dict[JointState, GameNode[Pr, X, U, Y, RP, RJ, SR]]
    depth: int
    gf: Optional[GameFactorization[X]]


def create_game_graph(
    game: Game[Pr, X, U, Y, RP, RJ, SR],
    dt: D,
    initials: AbstractSet[JointState],
    gf: Optional[GameFactorization[X]],
) -> GameGraph[Pr, X, U, Y, RP, RJ, SR]:
    state2node: Dict[JointState, GameNode[Pr, X, U, Y, RP, RJ, SR]] = {}
    ic = IterationContext(game, dt, state2node, depth=0, gf=gf)
    logger.info("creating game tree")
    for js in initials:
        create_game_graph_(ic, js)

    return GameGraph(initials, state2node)


def get_moves(
    ic: IterationContext[Pr, X, U, Y, RP, RJ, SR], js: JointState
) -> Mapping[PlayerName, Mapping[U, Poss[X, Pr]]]:
    """ Returns the possible moves. """
    res = {}
    state: X
    ps = ic.game.ps
    dt = ic.dt
    for player_name, state in js.items():
        player = ic.game.players[player_name]
        # is it a final state?
        is_final = player.personal_reward_structure.is_personal_final_state(state) if state else True

        if state is None or is_final:
            succ = {None: ps.lift_one(None)}
        else:
            succ = player.dynamics.successors(state, dt)
        res[player_name] = succ
    return res


def create_game_graph_(ic: IterationContext, js: JointState) -> GameNode[Pr, X, U, Y, RP, RJ, SR]:
    if js in ic.cache:
        return ic.cache[js]
    # logger.info(depth=ic.depth, js=js)
    states = {k: v for k, v in js.items() if v is not None}

    N2: JointState

    moves_to_state_everybody = get_moves(ic, js)
    pure_outcomes: Dict[JointPureActions, Poss[Mapping[PlayerName, JointState], Pr]] = {}
    ps = ic.game.ps
    ic2 = replace(ic, depth=ic.depth + 1)

    is_final = {}
    for player_name, player_state in states.items():
        _ = ic.game.players[player_name]
        if _.personal_reward_structure.is_personal_final_state(player_state):
            f = _.personal_reward_structure.personal_final_reward(player_state)
            is_final[player_name] = f

    who_exits = frozenset(ic.game.joint_reward.is_joint_final_state(states))
    joint_final = who_exits
    if joint_final:
        joint_final_rewards = ic.game.joint_reward.joint_reward(states)
    else:
        joint_final_rewards = {}

    players_exiting = set(who_exits) | set(is_final)

    # Consider only the moves of who remains
    not_exiting = lambda pn: pn not in players_exiting
    moves_to_state_remaining = fkeyfilter(not_exiting, moves_to_state_everybody)
    movesets_for_remaining = fvalmap(frozenset, moves_to_state_remaining)

    # Compute the incremental costs for the moves
    incremental = defaultdict(dict)
    for k, its_moves in moves_to_state_remaining.items():
        for move in its_moves:
            if move is None:
                continue
            pri = ic.game.players[k].personal_reward_structure.personal_reward_incremental
            inc = pri(states[k], move, ic.dt)
            incremental[k][move] = inc

    for joint_pure_action in iterate_dict_combinations(moves_to_state_remaining):

        pure_action: JointPureActions
        pure_action = fkeyfilter(lambda action: action is not None, joint_pure_action)

        if not pure_action:
            continue

        def f(item: Tuple[PlayerName, U]) -> Tuple[PlayerName, Poss[X, Pr]]:
            pn, choice = item
            return pn, moves_to_state_remaining[pn][choice]

        selected: Dict[PlayerName, Poss[X, Pr]]
        selected = itemmap(f, pure_action)

        def f(a: Mapping[PlayerName, U]) -> JointState:
            return fkeyfilter(not_exiting, a)

        outcomes: Poss[JointState, Pr] = ps.build_multiple(selected, f)

        # gt = create_game_graph_(ic2, N2)
        for _ in outcomes.support():
            if _:
                create_game_graph_(ic2, _)

        def r(js0: JointState) -> Mapping[PlayerName, JointState]:
            x = {k: js0 for k in states}
            return fkeyfilter(not_exiting, x)

        pure_outcomes[pure_action] = ps.build(outcomes, r)
        # pure_outcomes[pure_action] = ic.gp.game.ps.lift_one(gt.states)

    # moves = {k: frozenset(v) for k, v in moves.items()}
    # remove the fake moves {None}
    # for k, v in list(moves.items()):
    #     if v == {None}:
    #         moves.pop(k)
    # for player_name in joint_final:
    #     moves.pop(player_name, None)
    # for player_name in is_final:
    #     moves.pop(player_name, None)

    resources = {}
    for player_name, player_state in states.items():
        resources[player_name] = ic.game.players[player_name].dynamics.get_shared_resources(player_state)

    res = GameNode(
        moves=(movesets_for_remaining),
        states=frozendict(states),
        outcomes=frozendict(pure_outcomes),
        incremental=frozendict({k: frozendict(v) for k, v in incremental.items()}),
        joint_final_rewards=frozendict(joint_final_rewards),
        is_final=frozendict(is_final),
        resources=frozendict(resources),
    )
    ic.cache[js] = res
    return res
