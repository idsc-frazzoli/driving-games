import itertools
import random
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal as D
from functools import reduce
from itertools import product
from typing import (AbstractSet, Collection, Dict, FrozenSet as FSet, Generic, Iterator, List, Mapping, Set, Tuple,
                    TypeVar)

import numpy as np
from frozendict import frozendict
from networkx import connected_components, Graph, MultiDiGraph
from toolz import itemmap, valmap

from possibilities import check_poss, Poss, PossibilityStructure
from zuper_commons.types import ZException
from . import logger
from .game_def import (
    Dynamics,
    Game,
    GamePlayer,
    JointState,
    PersonalRewardStructure,
    PlayerName,
    Pr,
    RJ,
    RP,
    SR,
    U,
    X,
    Y,
)
from .single_game_tree import get_one_player_game_tree
from .structures_solution import (
    GameFactorization, GamePlayerPreprocessed,
    GamePreprocessed,
    GameSolution,
    SolvedGameNode,
    SolverParams,
    UsedResources,
)

__all__ = ["preprocess_game", "get_accessible_states"]


def preprocess_game(
    game: Game[Pr, X, U, Y, RP, RJ, SR],
    solver_params: SolverParams,
    individual: Mapping[PlayerName, Mapping[X, GameSolution[Pr, X, U, Y, RP, RJ]]],
) -> GamePreprocessed[Pr, X, U, Y, RP, RJ]:
    game_factorization: GameFactorization[X]
    game_factorization = get_game_factorization(game.ps, individual)
    game_graph = get_game_graph(game, dt=solver_params.dt)
    compute_graph_layout(game_graph, iterations=1)
    players_pre = {
        player_name: preprocess_player(game=game, player_name=player_name, player=player, dt=solver_params.dt)
        for player_name, player in game.players.items()
    }

    gp = GamePreprocessed(
        game=game, players_pre=players_pre, game_graph=game_graph, solver_params=solver_params,
        game_factorization=game_factorization
    )

    return gp



def get_game_factorization(
    ps: PossibilityStructure[Pr],
    individual: Mapping[PlayerName, Mapping[X, GameSolution[Pr, X, U, Y, RP, RJ]]],
) -> GameFactorization[X]:
    known: Mapping[PlayerName, Mapping[X, SolvedGameNode[Pr, X, U, Y, RP, RJ, SR]]]
    known = valmap(collapse_states, individual)
    js: JointState

    partitions: Dict[FSet[FSet[PlayerName]], Set[JointState]]
    partitions = defaultdict(set)
    ipartitions: Dict[JointState, FSet[FSet[PlayerName]]] = {}

    def get_ur(items: Tuple[PlayerName, X]) -> Tuple[PlayerName, UsedResources]:
        player_name, state = items
        return player_name, known[player_name][state].ur

    # iterate all combinations
    for js in iterate_dict_combinations(known):
        resources_used = itemmap(get_ur, js)
        independent = find_dependencies(ps, resources_used)
        partitions[independent].add(js)
        ipartitions[js] = independent

    mpartitions = valmap(frozenset, partitions)
    logger.info("stats", partitions=valmap(lambda _: len(_), partitions))
    return GameFactorization(mpartitions, ipartitions)


def find_dependencies(
    ps: PossibilityStructure[Pr], resources_used: Mapping[PlayerName, UsedResources[Pr, X, U, Y, RP, RJ, SR]]
) -> FSet[FSet[PlayerName]]:
    """
        Returns the dependency structure from the use of shared resources.
        Returns the partitions of players that are independent.

        Example: for 3 players '{a,b,c}' this could return  `{{a}, {b,c}}`.
        That means that `a` is independent
        of b and c. A return of  `{{a}, {b}, {c}}` means that all three are independent.
     """
    interaction_graph = Graph()
    interaction_graph.add_nodes_from(resources_used)
    max_instants = max(max(_.used) if _.used else 0 for _ in resources_used.values())
    for i in range(int(max_instants)):
        i = D(i)

        def getused(items) -> Tuple[PlayerName, FSet[SR]]:
            ur: UsedResources
            player_name, ur = items
            used: Mapping[D, Poss[Mapping[PlayerName, FSet[SR]], Pr]] = ur.used
            if i not in used:
                res = frozenset()
            else:
                at_i: Poss[Mapping[PlayerName, FSet[SR]], Pr] = ur.used[i]
                at_i_player: Poss[FSet[SR], Pr]
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

    return frozenset(map(frozenset, connected_components(interaction_graph)))


def flatten_sets(c: Collection[AbstractSet[X]]) -> FSet[X]:
    sets = reduce(lambda a, b: a | b, c)
    return frozenset(sets)


K = TypeVar("K")
V = TypeVar("V")


def iterate_dict_combinations(a: Mapping[K, Collection[V]]) -> Iterator[Mapping[K, V]]:
    ks = list(a)
    vs = [a[_] for _ in ks]
    alls = list(itertools.product(*tuple(vs)))
    for x in alls:
        d = frozendict(zip(ks, x))
        yield d


def collapse_states(
    sols: Mapping[X, GameSolution[Pr, X, U, Y, RP, RJ]]
) -> Mapping[X, SolvedGameNode[Pr, X, U, Y, RP, RJ, SR]]:
    res = {}
    for x, gs in sols.items():
        res.update(gs.states_to_solution)
    return res


def preprocess_player(
    game: Game, player_name: PlayerName, player: GamePlayer[Pr, X, U, Y, RP, RJ, SR], dt: D
):
    graph = get_player_graph(player, dt)
    alone_trees = {}
    for x0 in player.initial.support():
        alone_trees[x0] = get_one_player_game_tree(
            game=game, player_name=player_name, player=player, x0=x0, dt=dt
        )

    alone_trees = frozendict(alone_trees)
    return GamePlayerPreprocessed(graph, alone_trees)


def get_accessible_states(
    initial: Poss[X, Pr],
    personal_reward_structure: PersonalRewardStructure[X, U, RP],
    dynamics: Dynamics[Pr, X, U, SR],
    dt: D,
) -> MultiDiGraph:
    check_poss(initial, object)
    G = MultiDiGraph()

    for node in initial.support():
        i_final = personal_reward_structure.is_personal_final_state(node)
        if i_final:
            raise ZException(i_final=i_final)

        G.add_node(node, is_final=False)
    stack = list(initial.support())
    # logger.info(stack=stack)
    i: int = 0
    expanded = set()
    while stack:
        # print(i, len(stack), len(G.nodes))
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


def get_game_graph(game: Game[Pr, X, U, Y, RP, RJ, SR], dt: D) -> MultiDiGraph:
    players = game.players
    assert len(players) == 2
    p1, p2 = list(players)
    P1 = players[p1]
    P2 = players[p2]
    # G1 = get_player_graph(players[p1])
    # G2 = get_player_graph(players[p2])

    G = MultiDiGraph()
    stack: List[JointState] = []
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
    i = 0
    S: JointState
    ps = game.ps
    while stack:
        if i % 1000 == 0:
            logger.info("iteration", i=i, stack=len(stack), created=len(G.nodes))
        i += 1
        # S = stack.pop(0)
        S = stack.pop()
        assert S in G.nodes

        n1, n2 = S[p1], S[p2]

        if n1 is None or G.nodes[S]["is_final1"]:
            succ1 = {None: ps.lift_one(None)}
        else:
            succ1 = P1.dynamics.successors(n1, dt)

        if n2 is None or G.nodes[S]["is_final2"]:
            succ2 = {None: ps.lift_one(None)}
        else:
            succ2 = P2.dynamics.successors(n2, dt)

        generation = G.nodes[S]["generation"]

        for (u1, s1s), (u2, s2s) in product(succ1.items(), succ2.items()):
            check_poss(s1s, object)
            check_poss(s2s, object)
            for (s1, w1), (s2, w2) in product(s1s.it(), s2s.it()):
                # check_isinstance(s1, VehicleState)
                # check_isinstance(s2, VehicleState)
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
                    if not (is_joint_final):
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


def get_player_graph(player: GamePlayer[Pr, X, U, Y, RP, RJ, SR], dt: D) -> MultiDiGraph:
    return get_accessible_states(player.initial, player.personal_reward_structure, player.dynamics, dt=dt)
