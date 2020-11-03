import itertools
import random
from collections import defaultdict
from decimal import Decimal as D
from functools import reduce
from itertools import product
from typing import (
    AbstractSet,
    Collection,
    Dict,
    FrozenSet as FSet,
    List,
    Mapping,
    Set,
    Tuple,
    Optional,
)

import numpy as np
from frozendict import frozendict
from networkx import connected_components, Graph, MultiDiGraph
from toolz import itemmap, valmap

from possibilities import check_poss, Poss, PossibilityMonad
from zuper_commons.types import ZException
from . import logger
from .create_joint_game_tree import create_game_graph
from .game_def import (
    Dynamics,
    Game,
    GamePlayer,
    JointState,
    PersonalRewardStructure,
    PlayerName,
    RJ,
    RP,
    SR,
    U,
    X,
    Y,
)
from .get_indiv_games import get_individual_games
from .solution import solve_game2
from .structures_solution import (
    GameFactorization,
    GameGraph,
    GamePlayerPreprocessed,
    GamePreprocessed,
    GameSolution,
    SolvedGameNode,
    SolverParams,
    UsedResources,
)

__all__ = ["preprocess_game", "get_accessible_states"]

from .utils import fkeyfilter, iterate_dict_combinations


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
        lambda individual_game: preprocess_player(
            solver_params=solver_params, individual_game=individual_game
        ),
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


def get_game_factorization(
    game: Game[X, U, Y, RP, RJ, SR],
    players_pre: Mapping[PlayerName, GamePlayerPreprocessed[X, U, Y, RP, RJ, SR]],
) -> GameFactorization[X]:
    ps = game.ps
    known: Mapping[PlayerName, Mapping[JointState, SolvedGameNode[X, U, Y, RP, RJ, SR]]]
    known = valmap(collapse_states, players_pre)
    js: JointState

    partitions: Dict[FSet[FSet[PlayerName]], Set[JointState]]
    partitions = defaultdict(set)
    ipartitions: Dict[JointState, FSet[FSet[PlayerName]]] = {}

    def get_ur(items: Tuple[PlayerName, X]) -> Tuple[PlayerName, UsedResources]:
        pname, state = items
        return pname, known[pname][state].ur

    # iterate all combinations
    for ljs in iterate_dict_combinations(known):

        js_ = {}
        for player_name, joint_state_redundant in ljs.items():
            js_.update(joint_state_redundant)
        jsf = frozendict(js_)

        special = all(_.x == 0 for _ in jsf.values())
        # Note that if this is a final (collision) state, it is very important
        # that we do not consider it decoupled.. otherwise there is no collision
        # ever detected

        players_colliding = game.joint_reward.is_joint_final_state(jsf)
        if players_colliding:
            # logger.info('Found collision states', jsf=jsf, players_colliding=players_colliding)
            partition = frozenset({frozenset(players_colliding)})
            partitions[partition].add(jsf)
            ipartitions[jsf] = partition

            if special:
                logger.info(
                    "found that the players are colliding",
                    jsf=jsf,
                    players_colliding=players_colliding,
                    partition=partition,
                )
        else:
            resources_used = itemmap(get_ur, ljs)
            deps = find_dependencies(ps, resources_used)

            # if special:
            #     logger.info("the players are not colliding", jsf=jsf, resources_used=resources_used)
            for players_subsets, independent in deps.items():
                if special:
                    logger.info(" - ", players_subsets=players_subsets, independent=independent)
                jsf_subset = fkeyfilter(players_subsets.__contains__, jsf)
                partitions[independent].add(jsf_subset)
                ipartitions[jsf_subset] = independent

    # also for the single ones
    for player_name, player_states in known.items():
        for js in player_states:
            single = frozenset({frozenset({player_name})})
            partitions[single].add(js)
            ipartitions[js] = single

    mpartitions = valmap(frozenset, partitions)
    logger.info("stats", partitions=valmap(lambda _: len(_), partitions))
    return GameFactorization(mpartitions, ipartitions)


def find_dependencies(
    ps: PossibilityMonad,
    resources_used: Mapping[PlayerName, UsedResources[X, U, Y, RP, RJ, SR]],
) -> Mapping[FSet[PlayerName], FSet[FSet[PlayerName]]]:
    """
    Returns the dependency structure from the use of shared resources.
    Returns the partitions of players that are independent.

    Example: for 3 players '{a,b,c}' this could return  `{{a}, {b,c}}`.
    That means that `a` is independent
    of b and c. A return of  `{{a}, {b}, {c}}` means that all three are independent.

    For n players, it returns all combinations of subsets.
    """
    interaction_graph = Graph()
    interaction_graph.add_nodes_from(resources_used)
    max_instants = max(max(_.used) if _.used else 0 for _ in resources_used.values())
    for i in range(int(max_instants)):
        i = D(i)

        def getused(items) -> Tuple[PlayerName, FSet[SR]]:
            ur: UsedResources
            player_name, ur = items
            used: Mapping[D, Poss[Mapping[PlayerName, FSet[SR]]]] = ur.used
            if i not in used:
                res = frozenset()
            else:
                at_i: Poss[Mapping[PlayerName, FSet[SR]]] = ur.used[i]
                at_i_player: Poss[FSet[SR]]
                at_i_player = ps.build(at_i, lambda _: _[player_name])
                support_sets = flatten_sets(at_i_player.support())
                res = support_sets

            return player_name, res

        used_at_i = itemmap(getused, resources_used)

        p1: PlayerName
        p2: PlayerName
        for p1, p2 in itertools.combinations(resources_used, 2):
            intersection = used_at_i[p1] & used_at_i[p2]
            intersects = len(intersection) > 0
            if intersects:
                interaction_graph.add_edge(p1, p2)

    players = set(resources_used)
    n = len(players)
    result = {}
    for nplayers in range(2, n + 1):
        for players_subset in itertools.combinations(players, nplayers):
            G = interaction_graph.subgraph(players_subset)
            r = frozenset(map(frozenset, connected_components(G)))
            result[frozenset(players_subset)] = r
    return result


def flatten_sets(c: Collection[AbstractSet[X]]) -> FSet[X]:
    sets = reduce(lambda a, b: a | b, c)
    return frozenset(sets)


def collapse_states(
    gp: GamePlayerPreprocessed[X, U, Y, RP, RJ, SR]
) -> Mapping[JointState, SolvedGameNode[X, U, Y, RP, RJ, SR]]:
    return gp.gs.states_to_solution


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


def get_accessible_states(
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


def get_game_graph(game: Game[X, U, Y, RP, RJ, SR], dt: D) -> MultiDiGraph:
    players = game.players
    assert len(players) == 2
    p1, p2 = list(players)
    P1 = players[p1]
    P2 = players[p2]
    # G1 = get_player_graph(players[p1])
    # G2 = get_player_graph(players[p2])

    G = MultiDiGraph()
    stack: List[JointState] = []
    # root of the tree
    for n1, n2 in product(P1.initial.support(), P2.initial.support()):
        S = frozendict({p1: n1, p2: n2})
        G.add_node(
            S,
            is_final2=False,
            is_final1=False,
            is_joint_final=False,
            is_initial=True,
            generation=0,
            in_game="AB",
        )
        stack.append(S)
    logger.info(stack=stack)
    # all the rest of the tree
    i = 0
    S: JointState
    ps = game.ps
    while stack:
        if i % 1000 == 0:
            logger.info("iteration", i=i, stack=len(stack), created=len(G.nodes))
        i += 1
        S = stack.pop()
        assert S in G.nodes

        n1, n2 = S[p1], S[p2]

        if n1 is None or G.nodes[S]["is_final1"]:
            succ1 = {None: ps.unit(None)}
        else:
            succ1 = P1.dynamics.successors(n1, dt)

        if n2 is None or G.nodes[S]["is_final2"]:
            succ2 = {None: ps.unit(None)}
        else:
            succ2 = P2.dynamics.successors(n2, dt)

        generation = G.nodes[S]["generation"]

        for (u1, s1s), (u2, s2s) in product(succ1.items(), succ2.items()):
            check_poss(s1s, object)
            check_poss(s2s, object)
            for s1, s2 in product(s1s.support(), s2s.support()):
                if (s1, s2) == (None, None):
                    continue
                S2 = frozendict({p1: s1, p2: s2})
                if S2 not in G.nodes:
                    is_final1 = P1.personal_reward_structure.is_personal_final_state(s1) if s1 else True
                    is_final2 = P2.personal_reward_structure.is_personal_final_state(s2) if s2 else True

                    in_game = "AB" if (s1 and s2) else ("A" if s1 else "B")
                    if s1 and s2:
                        is_joint_final = len(game.joint_reward.is_joint_final_state({p1: s1, p2: s2})) > 0
                    else:
                        is_joint_final = False
                    G.add_node(
                        S2,
                        is_final2=is_final2,
                        is_final1=is_final1,
                        is_joint_final=is_joint_final,
                        is_initial=False,
                        generation=generation + 1,
                        in_game=in_game,
                    )
                    if not is_joint_final:
                        if S2 not in stack:
                            stack.append(S2)
                G.add_edge(S, S2, action=frozendict({p1: u1, p2: u2}))
                G.nodes[S2]["generation"] = min(G.nodes[S2]["generation"], generation + 1)
    return G


def compute_graph_layout(G: MultiDiGraph, iterations: int) -> None:
    generations = defaultdict(list)
    for n in G.nodes:
        g = G.nodes[n]["generation"]
        others = generations[g]
        others.append(n)

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
            in1 = ["A", "AB", "B"].index(in_game)
            return (in1, affinities[n_])

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
    return get_accessible_states(player.initial, player.personal_reward_structure, player.dynamics, dt=dt)
