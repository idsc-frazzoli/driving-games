import itertools
from collections import defaultdict
from decimal import Decimal as D
from functools import reduce
from typing import Dict, FrozenSet as FSet, Mapping, Set, Tuple, AbstractSet, Collection, Callable

from cytoolz import itemmap, valmap
from networkx import connected_components, Graph

from dg_commons import PlayerName, X, U, Y, RP, RJ, logger
from dg_commons.utils_toolz import iterate_dict_combinations, fkeyfilter, fd
from games.game_def import Game, JointState, SR
from games.solve.solution_structures import (
    GameFactorization,
    GamePlayerPreprocessed,
    SolvedGameNode,
    UsedResources,
)
from possibilities import PossibilityMonad, Poss


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
        return pname, known[pname][state].ur

    # iterate all combinations
    for ljs in iterate_dict_combinations(known):

        js_: Dict[PlayerName, JointState] = {}
        for player_name, joint_state_redundant in ljs.items():
            js_.update(joint_state_redundant)
        # fixme joint states factorized?!
        jsf = fd(js_)

        special = all(_.x == 0 for _ in jsf.values())  # fixme this can go?!
        # Note that if this is a final (collision) state, it is very important
        # that we do not consider it decoupled.. otherwise there is no collision ever detected

        # todo is this sufficient? now collisions are detected on transitions
        players_colliding = {}  # game.joint_reward.is_joint_final_states(jsf)

        if players_colliding:
            # logger.info('Found collision states', jsf=jsf, players_colliding=players_colliding)
            partition = frozenset({frozenset(players_colliding)})
            partitions[partition].add(jsf)
            ipartitions[jsf] = partition

            if special:
                logger.info(
                    "Found that the players are colliding",
                    jsf=jsf,
                    players_colliding=players_colliding,
                    partition=partition,
                )
            # todo need to add checks for the cases where one of the players has already finished?!
        else:
            resources_used = itemmap(get_ur, ljs)
            deps = find_dependencies(ps, resources_used, f_resource_intersection)

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
    f_resource_intersection: Callable[[FSet[SR], FSet[SR]], bool],
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

        def get_used(items) -> Tuple[PlayerName, FSet[SR]]:
            ur: UsedResources
            player_name, ur = items
            used: Mapping[D, Poss[Mapping[PlayerName, FSet[SR]]]] = ur.used
            if i not in used:
                res = frozenset()
            else:
                at_i: Poss[Mapping[PlayerName, FSet[SR]]] = ur.used[i]
                at_i_player: Poss[FSet[SR]]
                # todo  It could be that the player already finished for some actions (does not use any resources)
                #  and for some actions he didn't finish (uses resources) -> return default value empty set
                at_i_player = ps.build(at_i, lambda _: _[player_name])
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
