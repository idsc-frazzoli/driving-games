import itertools
from collections import defaultdict
from decimal import Decimal as D
from functools import reduce
from typing import AbstractSet, Callable, Collection, Dict, FrozenSet as FSet, Mapping, Set, Tuple

from cytoolz import itemmap, valmap
from networkx import connected_components, Graph

from dg_commons import logger, PlayerName, RJ, RP, U, X, Y
from dg_commons.utils_toolz import fd, fkeyfilter, iterate_dict_combinations
from games.game_def import Game, JointState, SR
from games.solve.solution_structures import (
    GameFactorization,
    GamePlayerPreprocessed,
    SolvedGameNode,
    UsedResources,
)
from possibilities import Poss, PossibilityMonad


def get_game_factorization(
    game: Game[X, U, Y, RP, RJ, SR],
    players_pre: Mapping[PlayerName, GamePlayerPreprocessed[X, U, Y, RP, RJ, SR]],
    f_resource_intersection: Callable[[FSet[SR], FSet[SR]], bool],
) -> GameFactorization[X]:
    """
    :param game:
    :param players_pre:
    :param f_resource_intersection:
    :return:
    """
    ps = game.ps
    known: Mapping[PlayerName, Mapping[JointState, SolvedGameNode[X, U, Y, RP, RJ, SR]]]
    known = valmap(collapse_states, players_pre)
    js: JointState

    partitions: Dict[FSet[FSet[PlayerName]], Set[JointState]]
    partitions = defaultdict(set)
    ipartitions: Dict[JointState, FSet[FSet[PlayerName]]] = {}

    def get_ur(items: Tuple[PlayerName, X]) -> Tuple[PlayerName, UsedResources]:
        pname, state = items
        return pname, known[pname][state].reachable_res

    # iterate all combinations
    for ljs in iterate_dict_combinations(known):

        js_: Dict[PlayerName, JointState] = {}
        for player_name, joint_state_redundant in ljs.items():
            js_.update(joint_state_redundant)
        # fixme joint states factorized?! skip if the state has been already factorized
        jsf = fd(js_)

        resources_used = itemmap(get_ur, ljs)
        deps = find_dependencies(ps, resources_used, f_resource_intersection)

        for players_subsets, independent in deps.items():
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


# bug: need to also have reachable resources
# then can check with optimal resources if single-player game
# otherwise need to use reachable
def find_dependencies(
    ps: PossibilityMonad,
    resources_used: Mapping[PlayerName, UsedResources[X, U, Y, RP, RJ, SR]],
    f_resource_intersection: Callable[[FSet[SR], FSet[SR]], bool],
) -> Mapping[FSet[PlayerName], FSet[FSet[PlayerName]]]:
    """
    Returns the dependency structure from the use of shared resources.
    Returns the partitions of players that are independent.

    Example: for 3 players '{a,b,c}' this could return  `{{a}, {b,c}}`.
    That means that `a` is independent  of b and c.
    A return of  `{{a}, {b}, {c}}` means that all three are independent.
    For n players, it returns all combinations of subsets.
    """
    interaction_graph = Graph()
    interaction_graph.add_nodes_from(resources_used)
    max_stages = max(max(_.used) if _.used else 0 for _ in resources_used.values())
    for i in range(int(max_stages)):
        i = D(i)

        def get_used(items) -> Tuple[PlayerName, FSet[SR]]:
            ur: UsedResources
            player_name, ur = items
            used: Mapping[D, Poss[Mapping[PlayerName, FSet[SR]]]] = ur.used
            if i not in used:
                res = frozenset()
            else:
                at_i: Poss[Mapping[PlayerName, FSet[SR]]] = ur.used[i]
                at_i_player: Poss[FSet[SR]]
                # It could be that the player already finished for some actions (does not use any resources)
                # -> return default value empty set
                at_i_player = ps.build(at_i, lambda _: _.get(player_name, frozenset()))
                support_sets = flatten_sets(at_i_player.support())
                res = support_sets

            return player_name, res

        used_at_i: Mapping[PlayerName, FSet[SR]] = itemmap(get_used, resources_used)

        p1: PlayerName
        p2: PlayerName
        for p1, p2 in itertools.combinations(resources_used, 2):
            intersects = f_resource_intersection(used_at_i[p1], used_at_i[p2])
            if intersects:
                interaction_graph.add_edge(p1, p2)

    players = set(resources_used)
    n = len(players)
    result = {}
    # fixme this last part is not necessary if we are factorizing while building the game tree?!
    #   or we need to store the results somewhere
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
