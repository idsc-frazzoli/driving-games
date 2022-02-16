from collections import defaultdict
from dataclasses import dataclass, replace
from decimal import Decimal as D
from typing import AbstractSet, Callable, Dict, FrozenSet as FSet, Generic, Mapping, Set, Tuple

from networkx import DiGraph, topological_sort
from toolz import itemmap

from dg_commons import DgSampledSequence, PlayerName, RJ, RP, U, X, Y
from dg_commons.utils_toolz import *
from possibilities import Poss
from . import logger
from .checks import check_joint_state
from .factorization import collapse_states, find_dependencies
from .game_def import Combined, Game, JointPureActions, JointState, SR
from .solve.solution_structures import (
    AccessibilityInfo,
    GameGraph,
    GameNode,
    GamePlayerPreprocessed,
    SolvedGameNode,
    UsedResources,
)


@dataclass
class IterationContextFact(Generic[X, U, Y, RP, RJ, SR]):
    """Iteration structure while creating the game graph."""

    game: Game[X, U, Y, RP, RJ, SR]
    """The game."""
    dt: D
    """ Solver discretization time """
    cache: Dict[JointState, GameNode[X, U, Y, RP, RJ, SR]]
    """ Nodes that were already computed. """
    depth: int
    """ The current depth. """
    known: Mapping[PlayerName, Mapping[JointState, SolvedGameNode[X, U, Y, RP, RJ, SR]]]
    """Known preprocessed players"""
    f_resource_intersection: Callable[[FSet[SR], FSet[SR]], bool]
    """Function to check for intersection of resources"""


def create_game_graph_fact(
    game: Game[X, U, Y, RP, RJ, SR],
    dt: D,
    initials: AbstractSet[JointState],
    players_pre: Mapping[PlayerName, GamePlayerPreprocessed[X, U, Y, RP, RJ, SR]],
    f_resource_intersection: Callable[[FSet[SR], FSet[SR]], bool],
) -> GameGraph[X, U, Y, RP, RJ, SR]:
    """Create the game graph checking for factorization at the same time."""
    state2node: Dict[JointState, GameNode[X, U, Y, RP, RJ, SR]] = {}
    known: Mapping[PlayerName, Mapping[JointState, SolvedGameNode[X, U, Y, RP, RJ, SR]]]
    known = valmap(collapse_states, players_pre)
    ic = IterationContextFact(
        game, dt, state2node, depth=0, known=known, f_resource_intersection=f_resource_intersection
    )

    logger.info("Creating game tree")
    # todo ideally one could check if the initial state can be already factorized
    for js in initials:
        _create_game_graph_fact(ic, js)

    logger.info(f"Created game graph with {len(state2node)} game nodes")

    # create networkx graph
    G = get_networkx_graph(state2node)
    ti = get_timestep_info(G, dt)

    # visualize number of states by time
    sizes = {}
    for t, states in ti.time2states.items():
        res = defaultdict(lambda: 0)
        for js in states:
            res[len(js)] += 1
        sizes[t] = dict(sorted(res.items()))
    logger.info("Number of states by time", sizes=sizes)

    return GameGraph(initials, state2node, ti)


def get_timestep_info(G: DiGraph, dt: D) -> AccessibilityInfo[X]:
    """Computes which states are reachable at what time."""
    state2times: Dict[JointState, Set[D]] = defaultdict(set)
    time2states: Dict[D, Set[JointState]] = defaultdict(set)

    # traverse in topological sort
    ts = list(topological_sort(G))
    for n1 in ts:
        # if first time
        if n1 not in state2times:
            # it is at time 0
            state2times[n1].add(D(0))
            time2states[D(0)].add(n1)
        # for all its successors
        for n2 in G.successors(n1):
            # for each time t1 at which we can be at n1
            for t1 in state2times[n1]:
                # we can be at n2 at time t2
                t2 = t1 + dt
                state2times[n2].add(t2)
                time2states[t2].add(n2)
    return AccessibilityInfo(state2times, time2states)


def get_networkx_graph(state2node: Dict[JointState, GameNode[X, U, Y, RP, RJ, SR]]) -> DiGraph:
    """Returns a NetworkX DiGraph that summarizes the relation of the nodes."""
    G = DiGraph()
    G.add_nodes_from(state2node)
    for js, gn in state2node.items():
        for p in gn.transitions.values():
            for d in p.support():
                for _, js2 in d.items():
                    G.add_edge(js, js2)
    return G


def get_moves(
    ic: IterationContextFact[X, U, Y, RP, RJ, SR], js: JointState
) -> Mapping[PlayerName, Mapping[U, Poss[X]]]:
    """Returns the possible moves and the corresponding possible future states."""
    res = {}
    state: X
    ps = ic.game.ps
    dt = ic.dt
    for player_name, state in js.items():
        player = ic.game.players[player_name]
        # is it a final state?
        is_final = player.personal_reward_structure.is_personal_final_state(state) if state else True
        # todo check if we also need to add the check on the state collided

        if state is None or is_final:
            succ = {None: ps.unit(None)}
        else:
            succ = player.dynamics.successors(state, dt)
        res[player_name] = succ
    return res


def _create_game_graph_fact(ic: IterationContextFact, states: JointState) -> GameNode[X, U, Y, RP, RJ, SR]:
    """
    Builds a game node from the joint state.
    :param ic:
    :param states:
    :return:
    """
    check_joint_state(states)
    if states in ic.cache:
        return ic.cache[states]

    moves_to_state_everybody = get_moves(ic, states)
    pure_transitions: Dict[JointPureActions, Poss[Mapping[PlayerName, JointState]]] = {}
    pure_incremental: Dict[JointPureActions, Poss[Mapping[PlayerName, Combined]]] = {}
    ps = ic.game.ps
    ic2 = replace(ic, depth=ic.depth + 1)

    is_personal_final = {}
    for player_name, player_state in states.items():
        _ = ic.game.players[player_name]
        if _.personal_reward_structure.is_personal_final_state(player_state):
            f = _.personal_reward_structure.personal_final_reward(player_state)
            is_personal_final[player_name] = f
    who_has_collided = frozenset(ic.game.joint_reward.is_joint_final_states(states))
    joint_final_rewards = ic.game.joint_reward.joint_final_reward(states)

    players_exiting = set(who_has_collided) | set(is_personal_final)
    # Consider only the moves of whom remains
    not_exiting = lambda pn: pn not in players_exiting
    moves_to_state_remaining = fkeyfilter(not_exiting, moves_to_state_everybody)
    movesets_for_remaining = fvalmap(frozenset, moves_to_state_remaining)

    # Compute the personal incremental costs for the moves
    pers_incremental_cost: Mapping[PlayerName, Dict[U, Poss[RP]]] = defaultdict(dict)
    for k, its_moves in moves_to_state_remaining.items():
        pri = ic.game.players[k].personal_reward_structure.personal_reward_incremental
        for move in its_moves:
            if move is None:
                continue
            inc = pri(states[k], move, ic.dt)
            pers_incremental_cost[k][move] = inc

    # compute next joint states and incremental joint rewards
    for joint_pure_action in iterate_dict_combinations(moves_to_state_remaining):
        pure_action: JointPureActions = fkeyfilter(lambda action: action is not None, joint_pure_action)
        if not pure_action:
            continue

        def f(item: Tuple[PlayerName, U]) -> Tuple[PlayerName, Poss[X]]:
            pn, choice = item
            return pn, moves_to_state_remaining[pn][choice]

        selected: Dict[PlayerName, Poss[X]]
        selected = itemmap(f, pure_action)

        def f(a: Mapping[PlayerName, U]) -> JointState:
            return fkeyfilter(not_exiting, a)

        next_states: Poss[JointState] = ps.build_multiple(selected, f)

        # here compute the joint rewards (consider the non factorized next_state for the current stage cost)
        def transition_cost(_next_state: JointState) -> Mapping[PlayerName, Combined]:
            transitions = {
                p: DgSampledSequence[X](timestamps=(D(0), ic.dt), values=(states[p], _next_state[p]))
                for p in _next_state
            }
            m_pn_rj = ic.game.joint_reward.joint_reward_incremental(transitions)
            return fd(
                {p: Combined(personal=pers_incremental_cost[p][pure_action[p]], joint=m_pn_rj[p]) for p in _next_state}
            )

        trans_cost: Poss[Mapping[PlayerName, Combined]] = ps.build(next_states, transition_cost)
        pure_incremental[joint_pure_action] = trans_cost

        # here the generalized transition to support factorization
        def r(js0: JointState) -> Mapping[PlayerName, JointState]:
            js_continuing = fkeyfilter(not_exiting, js0)
            fact_states: Dict[PlayerName, JointState] = {}

            if len(js_continuing) > 1:

                def get_ur(items: Tuple[PlayerName, X]) -> Tuple[PlayerName, UsedResources]:
                    pname, state = items
                    alone_js = fd({pname: state})
                    return pname, ic.known[pname][alone_js].ur

                resources_used = itemmap(get_ur, js_continuing)
                deps: Mapping[FSet[PlayerName], FSet[FSet[PlayerName]]]

                # bug: we cannot use the optimal resources to factorize sets of >1 players

                deps = find_dependencies(ps, resources_used, ic.f_resource_intersection)

                pset: FSet[PlayerName]
                for pset in deps[frozenset(js_continuing)]:
                    jsf: JointState = fd({p: js0[p] for p in pset})
                    for p in pset:
                        fact_states[p] = jsf
                # logger.info(deps=deps, fact_states=fact_states)
            else:
                for p in js_continuing:
                    fact_states[p] = js0
            return fd(fact_states)

        pnext_states: Poss[Mapping[PlayerName, JointState]] = ps.build(next_states, r)

        # here need to update who has collided (their state in next_states)(to be done after factorization)
        collided_in_transition = {pn for tc in trans_cost.support() for pn in tc if tc[pn].joint.collision is not None}
        if collided_in_transition:

            def update_states_collided(pjs: Mapping[PlayerName, JointState]) -> Mapping[PlayerName, JointState]:
                def update_js(js: JointState) -> JointState:
                    js = dict(js)
                    for pn in js:
                        if pn in collided_in_transition:
                            js[pn] = replace(js[pn], has_collided=True)
                    return fd(js)

                js_new = valmap(update_js, pjs)
                return fd(js_new)

            pnext_states = ps.build(pnext_states, update_states_collided)

        pure_transitions[pure_action] = pnext_states

        for pn in pnext_states.support():
            for _, js_ in pn.items():
                _create_game_graph_fact(ic2, js_)

    resources = {}
    for player_name, player_state in states.items():
        dynamics = ic.game.players[player_name].dynamics
        resources[player_name] = dynamics.get_shared_resources(player_state, ic.dt)

    res = GameNode(
        states=fd(states),
        moves=movesets_for_remaining,
        transitions=fd(pure_transitions),
        personal_final_reward=fd(is_personal_final),
        incremental=fd(pure_incremental),
        joint_final_rewards=joint_final_rewards,
        resources=fd(resources),
    )
    ic.cache[states] = res
    return res
