import random
from collections import defaultdict
from decimal import Decimal as D
from typing import Dict, List, Mapping, Set, Optional, NoReturn

import numpy as np
from cytoolz import valmap
from frozendict import frozendict
from networkx import MultiDiGraph
from zuper_commons.types import ZException

from dg_commons import DgSampledSequence, PlayerName, X, U, Y, RP, RJ
from dg_commons.time import time_function
from dg_commons.utils_toolz import iterate_dict_combinations
from games import logger
from games.create_joint_game_tree import create_game_graph
from games.factorization import get_game_factorization
from games.game_def import (
    Dynamics,
    Game,
    GamePlayer,
    JointState,
    PersonalRewardStructure,
    SR,
)
from games.get_indiv_games import get_individual_games
from games.solve.solution import solve_game2
from games.solve.solution_structures import (
    GameFactorization,
    GameGraph,
    GamePlayerPreprocessed,
    GamePreprocessed,
    GameSolution,
    SolverParams,
)
from possibilities import check_poss, Poss

__all__ = ["preprocess_game", "get_reachable_states"]


def preprocess_game(
    game: Game[X, U, Y, RP, RJ, SR],
    solver_params: SolverParams,
) -> GamePreprocessed[X, U, Y, RP, RJ, SR]:
    """
    1. Preprocesses the game computing the general game graph (MultiDiGraph used for visualisation)
    2. Computes the solutions for the single players
    3. If factorization is selected, computes the corresponding game factorization

    :param game:
    :param solver_params:
    :return:
    """
    game_factorization: Optional[GameFactorization[X]] = None

    game_graph = get_game_graph(game, dt=solver_params.dt)
    compute_graph_layout(game_graph, iterations=1)
    individual_games = get_individual_games(game)
    players_pre = valmap(
        lambda individual_game: preprocess_player(solver_params=solver_params, individual_game=individual_game),
        individual_games,
    )
    if solver_params.use_factorization:
        game_factorization = get_game_factorization(game, players_pre)

    gp = GamePreprocessed(
        game=game,
        players_pre=players_pre,
        game_graph=game_graph,
        solver_params=solver_params,
        game_factorization=game_factorization,
    )

    return gp


def preprocess_player(
    individual_game: Game[X, U, Y, RP, RJ, SR],
    solver_params: SolverParams,
) -> GamePlayerPreprocessed[X, U, Y, RP, RJ, SR]:
    """
    # todo
    :param individual_game:
    :param solver_params:
    :return:
    """
    l = list(individual_game.players)
    assert len(l) == 1
    player_name = l[0]
    player: GamePlayer = individual_game.players[player_name]
    graph = get_player_graph(player, solver_params.dt)

    game_graph: GameGraph[X, U, Y, RP, RJ, SR]
    initials = frozenset(map(lambda x: frozendict({player_name: x}), player.initial.support()))

    game_graph = create_game_graph(individual_game, solver_params.dt, initials, gf=None)

    gs: GameSolution[X, U, Y, RP, RJ, SR]
    gs = solve_game2(game=individual_game, solver_params=solver_params, gg=game_graph, jss=initials)

    return GamePlayerPreprocessed(graph, game_graph, gs)


def get_reachable_states(
    initial: Poss[X],
    personal_reward_structure: PersonalRewardStructure[X, U, RP],
    dynamics: Dynamics[X, U, SR],
    dt: D,
) -> MultiDiGraph:
    """
    Computes the states accessible for a player subject to their dynamics and their personal cost function.
    :param initial:
    :param personal_reward_structure:
    :param dynamics:
    :param dt:
    :return:
    """
    check_poss(initial, object)
    G = MultiDiGraph()

    for node in initial.support():
        i_final = personal_reward_structure.is_personal_final_state(node)
        if i_final:
            raise ZException(i_final=i_final)
        G.add_node(node, is_final=False)

    stack = list(initial.support())
    i: int = 0
    expanded = set()
    while stack:
        i += 1
        s1 = stack.pop(0)
        assert s1 in G.nodes
        if s1 in expanded:
            continue
        # is_final =  player.personal_reward_structure.is_personal_final_state(s1)
        # G.add_node(s1, is_final=is_final)
        # # logger.info(s1=G.nodes[s1])

        expanded.add(s1)
        successors = dynamics.successors(s1, dt)
        for u, p_s2 in successors.items():
            check_poss(p_s2, object)
            # for u, s2s in successors.items():
            for s2 in p_s2.support():
                if s2 not in G.nodes:
                    is_final2 = personal_reward_structure.is_personal_final_state(s2)
                    G.add_node(s2, is_final=is_final2)
                    if not is_final2:
                        stack.append(s2)

                G.add_edge(s1, s2, u=u)
    return G


@time_function
def get_game_graph(game: Game[X, U, Y, RP, RJ, SR], dt: D) -> MultiDiGraph:
    """Gets the game graph, ?used only for visualisation? the real game is built in create_joint_game_tree"""
    players = game.players
    init_states: Mapping[PlayerName, X] = valmap(lambda x: x.initial.support(), players)

    G = MultiDiGraph()
    stack: List[JointState] = []
    # root of the tree
    for S in iterate_dict_combinations(init_states):
        G.add_node(
            S,
            is_final_for=[],
            is_joint_final=False,
            is_initial=True,
            generation=0,
            in_game="-".join(S.keys()),
        )
        stack.append(S)
    logger.info(stack=stack)
    # all the rest of the tree
    i = 0
    S: JointState
    ps = game.ps
    while stack:
        if i % 1000 == 0:
            logger.info("Iteration", i=i, stack=len(stack), created=len(G.nodes))
        i += 1
        S = stack.pop()
        assert S in G.nodes

        players_alive = filter(lambda x: x not in G.nodes[S]["is_final_for"], S)
        successors: Dict[PlayerName : Mapping[U, Poss[X]]] = {}
        for p in players_alive:
            p_state = S[p]
            p_succs = players[p].dynamics.successors(p_state, dt)
            successors[p] = p_succs

        generation = G.nodes[S]["generation"]
        for players_n_actions in iterate_dict_combinations(successors):
            poss_next = [successors[p][action].support() for p, action in players_n_actions.items()]
            players_poss_next = dict(zip(players_n_actions, poss_next))
            for S2 in iterate_dict_combinations(players_poss_next):
                if S2 not in G.nodes:
                    ending_players = [
                        p for p in S2 if players[p].personal_reward_structure.is_personal_final_state(S2[p])
                    ]
                    transitions = {p: DgSampledSequence[X](timestamps=(D(0), dt), values=(S[p], S2[p])) for p in S2}
                    jointly_ending = game.joint_reward.is_joint_final_transition(transitions)
                    ending_players.extend(jointly_ending)
                    is_joint_final = len(jointly_ending) > 0
                    G.add_node(
                        S2,
                        is_final_for=ending_players,
                        is_joint_final=is_joint_final,
                        is_initial=False,
                        generation=generation + 1,
                        in_game="-".join(S2.keys()),
                    )
                    # if anyone is still alive add to stack for further expansion
                    if any(p not in ending_players for p in S2):
                        if S2 not in stack:
                            stack.append(S2)
                G.add_edge(S, S2, action=players_n_actions)
                G.nodes[S2]["generation"] = min(G.nodes[S2]["generation"], generation + 1)
    logger.info("Game nodes", created=len(G.nodes))
    return G


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


def get_player_graph(player: GamePlayer[X, U, Y, RP, RJ, SR], dt: D) -> MultiDiGraph:
    return get_reachable_states(player.initial, player.personal_reward_structure, player.dynamics, dt=dt)
