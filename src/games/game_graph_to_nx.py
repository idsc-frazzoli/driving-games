import random
from collections import defaultdict
from itertools import product, chain
from typing import Mapping, List, Tuple, Set, NoReturn

import numpy as np
from frozendict import frozendict
from networkx import MultiDiGraph, simple_cycles
from zuper_commons.types import ZValueError

from dg_commons import PlayerName, U, X, Y, RP, RJ
from games import logger, JointState, SR
from games.checks import check_joint_state
from games.solve.solution_structures import GameGraph, GameNode
from possibilities import Poss

__all__ = ["build_networkx_from_game_graph", "compute_graph_layout"]


def build_networkx_from_game_graph(gg: GameGraph[X, U, Y, RP, RJ, SR]) -> MultiDiGraph:
    """Creates a networkx version of the game graph, currently used only for visualisation."""
    stack: List[Tuple[int, JointState]] = list(product([0], gg.initials))
    G = MultiDiGraph()

    i = 0
    js: JointState
    while stack:
        if i % 1000 == 0:
            logger.info("Iteration", i=i, stack=len(stack), created=len(G.nodes))

        generation, js = stack.pop()
        if js not in G.nodes:
            assert generation == 0, (generation, js)
            _add_node(G, js, state2node=gg.state2node, generation=generation)
        pjs_: Poss[Mapping[PlayerName, JointState]]
        if js not in gg.state2node:
            raise ZValueError("State not in state2node", js=js, state2node=gg.state2node.keys())
        for joint_actions, pjs_ in gg.state2node[js].transitions.items():
            js2: JointState
            next_states: Set[JointState] = set(chain.from_iterable(map(lambda x: set(x.values()), pjs_.support())))
            for js2 in next_states:
                check_joint_state(js2)
                if js2 not in G.nodes:
                    _add_node(G, js2, state2node=gg.state2node, generation=generation + 1)
                    stack.append((generation + 1, js2))
                G.add_edge(js, js2, action=joint_actions)
        i += 1
    logger.info(f"Networkx graph has {len(G.nodes)}")
    cycles = list(simple_cycles(G))
    if cycles:
        msg = "Did not expect cycles in the graph"
        raise ZValueError(msg, cycles=cycles)
    return G


def _add_node(G: MultiDiGraph, js: JointState, state2node: Mapping[JointState, GameNode], generation: int):
    gnode: GameNode = state2node[js]
    G.add_node(
        js,
        is_joint_final_for="-".join(gnode.joint_final_rewards),
        is_pers_final="-".join(gnode.personal_final_reward),
        is_initial=generation == 0,
        is_terminal=gnode.moves == frozendict({}),
        generation=generation,
        in_game="-".join(js.keys()),
        n_players=len(js),
    )
    return


def compute_graph_layout(G: MultiDiGraph, iterations: int) -> NoReturn:
    """Sorts nodes for visualisation"""
    generations = defaultdict(list)
    games_sets: Set[str] = set()
    for n in G.nodes:
        games_sets.add(G.nodes[n]["in_game"])
        g = G.nodes[n]["generation"]
        others = generations[g]
        others.append(n)

    sorted_games = sorted(games_sets)
    logger.info("reordering")

    for it in range(iterations):
        g = random.choice(list(generations))
        if g == 0 or g == len(generations) - 1:
            continue
        ordered = generations[g]
        affinities = {}
        for n in ordered:
            pred_order = []
            for p in G.predecessors(n):
                p_g = G.nodes[p]["generation"]
                if p_g != g - 1:
                    continue
                pred_order.append(generations[p_g].index(p))
            for s in G.successors(n):
                s_g = G.nodes[s]["generation"]
                if s_g != g + 1:
                    continue
                pred_order.append(generations[s_g].index(s))
            if not pred_order:
                val = 0  # should it happen?
            else:
                val = (np.mean(pred_order) + np.median(pred_order)) / 2
            # val =  np.mean(pred_order)
            # val =  np.median(pred_order)
            affinities[n] = val

        def ordering(n_):
            in_game = G.nodes[n_]["in_game"]
            in1 = sorted_games.index(in_game)
            return in1, affinities[n_]

        reordered = sorted(ordered, key=ordering)

        nchanges = []
        for i, _ in enumerate(ordered):
            j = reordered.index(_)
            diff = np.abs(i - j)
            nchanges.append(diff)
        avg = np.mean(nchanges)
        logger.info(f"iteration {it:4} {g:4} {avg:10.2f}")
        generations[g] = reordered

    for n in G.nodes:
        g = G.nodes[n]["generation"]
        others = generations[g]
        N = float(len(others))
        G.nodes[n]["y"] = (float(others.index(n) - N / 2) / N) * 400
        G.nodes[n]["x"] = g * 200
