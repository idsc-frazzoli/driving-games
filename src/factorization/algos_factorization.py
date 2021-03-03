import itertools
from time import perf_counter
from collections import defaultdict
from decimal import Decimal as D
from typing import (
    Dict,
    FrozenSet as FSet,
    Mapping,
    Set,
    Tuple,
    Optional,
)

from frozendict import frozendict
from networkx import connected_components, Graph
from toolz import itemmap, valmap

from possibilities import Poss, PossibilityMonad
from games import logger
from games.game_def import (
    Game,
    JointState,
    PlayerName,
    RJ,
    RP,
    SR,
    U,
    X,
    Y,
)
from games.utils import fkeyfilter, iterate_dict_combinations
from games.performance import GetFactorizationPI
from games.solve.solution_structures import (
    GameFactorization,
    GamePlayerPreprocessed,
    SolvedGameNode,
    UsedResources,
)
from games.access import collapse_states, flatten_sets


def get_game_factorization_no_collision_check(
    game: Game[X, U, Y, RP, RJ, SR],
    players_pre: Mapping[PlayerName, GamePlayerPreprocessed[X, U, Y, RP, RJ, SR]],
    fact_perf: Optional[GetFactorizationPI] = None
) -> GameFactorization[X]:
    """

    :param game:
    :param players_pre:
    :param fact_perf:
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

        js_ = {}
        for player_name, joint_state_redundant in ljs.items():
            js_.update(joint_state_redundant)
        jsf = frozendict(js_)

        special = all(_.x == 0 for _ in jsf.values())

        resources_used = itemmap(get_ur, ljs)

        # start timer to collect the time for finding dependencies between players
        t1 = perf_counter()

        deps = find_dependencies_no_collision_check(ps, resources_used)

        # stop timer and collect performance if given
        t2 = perf_counter()
        tot_t = t2 - t1
        if fact_perf:
            fact_perf.total_time_find_dependencies += tot_t

        # if special:
        #     logger.info("the players are not colliding", jsf=jsf, resources_used=resources_used)
        for players_subsets, independent in deps.items():
            # if special:
            #     logger.info(" - ", players_subsets=players_subsets, independent=independent)
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
    # logger.info("stats", partitions=valmap(lambda _: len(_), partitions))
    return GameFactorization(mpartitions, ipartitions)


def find_dependencies_no_collision_check(
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
    for i in range(int(max_instants) + 1):
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
                # It could be that the player already finished for some actions (does not use any resources)
                # and for some actions he didn't finish (uses resources) -> return default value empty set
                at_i_player = ps.build(at_i, lambda _: _.get(player_name, frozenset()))
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
