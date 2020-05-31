from collections import defaultdict
from dataclasses import dataclass, replace
from decimal import Decimal as D
from typing import AbstractSet, Dict, Generic, Mapping, Optional, Set, Tuple

from frozendict import frozendict
from networkx import DiGraph, topological_sort
from toolz import itemmap

from possibilities import Poss
from zuper_commons.types import ZValueError
from . import logger
from .game_def import (
    check_joint_state,
    Game,
    JointPureActions,
    JointState,
    PlayerName,
    Pr,
    RJ,
    RP,
    SR,
    U,
    X,
    Y,
)
from .structures_solution import AccessibilityInfo, GameFactorization, GameGraph, GameNode
from .utils import fkeyfilter, fvalmap, iterate_dict_combinations

__all__ = []


@dataclass
class IterationContext(Generic[Pr, X, U, Y, RP, RJ, SR]):
    game: Game[Pr, X, U, Y, RP, RJ, SR]
    dt: D
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

    # create networkx graph
    G = get_networkx_graph(state2node)
    ti = get_timestep_info(G)
    # time2nstates = valmap(len, ti.time2states)
    # logger.info('States accessible at each time', time2nstates=time2nstates)
    sizes = {}
    for t, states in ti.time2states.items():
        res = defaultdict(lambda: 0)
        for js in states:
            res[len(js)] += 1
        sizes[t] = dict(sorted(res.items()))
    logger.info("Number of states by time", sizes=sizes)

    return GameGraph(initials, state2node, ti)


def get_timestep_info(G: DiGraph) -> AccessibilityInfo[X]:
    ts = list(topological_sort(G))
    # logger.info(ts=ts)
    state2times: Dict[JointState, Set[D]] = defaultdict(set)
    time2states: Dict[D, Set[JointState]] = defaultdict(set)
    for n1 in ts:
        if n1 not in state2times:
            state2times[n1].add(D(0))
            time2states[D(0)].add(n1)
        for n2 in G.successors(n1):
            for t1 in state2times[n1]:
                t2 = t1 + D(1)
                state2times[n2].add(t2)
                time2states[t2].add(n2)
    return AccessibilityInfo(state2times, time2states)


def get_networkx_graph(state2node: Dict[JointState, GameNode[Pr, X, U, Y, RP, RJ, SR]]):
    G = DiGraph()
    G.add_nodes_from(state2node)
    for js, gn in state2node.items():
        for p in gn.outcomes.values():
            for d in p.support():
                for _, js2 in d.items():
                    G.add_edge(js, js2)
    return G


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


def create_game_graph_(ic: IterationContext, states: JointState) -> GameNode[Pr, X, U, Y, RP, RJ, SR]:
    check_joint_state(states)
    if states in ic.cache:
        return ic.cache[states]

    moves_to_state_everybody = get_moves(ic, states)
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

        def r(js0: JointState) -> Mapping[PlayerName, JointState]:
            if ic.gf is not None:
                # using game factorization
                js_continuing = fkeyfilter(not_exiting, js0)
                if js_continuing not in ic.gf.ipartitions:
                    msg = "Cannot find the state in the factorization info"
                    raise ZValueError(msg, js0=js_continuing, known=set(ic.gf.ipartitions))
                partitions = ic.gf.ipartitions[js_continuing]
                re = {}
                for players_in_partition in partitions:
                    this_partition_state = fkeyfilter(lambda pn: pn in players_in_partition, js_continuing)
                    for pname in players_in_partition:
                        re[pname] = this_partition_state
                return frozendict(re)
            else:
                x = {k_: js0 for k_ in states}
                return fkeyfilter(not_exiting, x)

        poutcomes: Poss[Mapping[PlayerName, JointState], Pr] = ps.build(outcomes, r)
        pure_outcomes[pure_action] = poutcomes

        for p in poutcomes.support():
            for _, js_ in p.items():
                create_game_graph_(ic2, js_)

    resources = {}
    for player_name, player_state in states.items():
        dynamics = ic.game.players[player_name].dynamics
        resources[player_name] = dynamics.get_shared_resources(player_state)

    res = GameNode(
        moves=(movesets_for_remaining),
        states=frozendict(states),
        outcomes=frozendict(pure_outcomes),
        incremental=fvalmap(frozendict, incremental),
        joint_final_rewards=frozendict(joint_final_rewards),
        is_final=frozendict(is_final),
        resources=frozendict(resources),
    )
    ic.cache[states] = res
    return res
