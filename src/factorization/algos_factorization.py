import itertools
from time import perf_counter
from collections import defaultdict
from decimal import Decimal as D
from copy import deepcopy
import math

from typing import (
    Dict,
    FrozenSet as FSet,
    Mapping,
    Set,
    Tuple,
    Optional,
    Iterator,
    List
)

from dataclasses import replace
from frozendict import frozendict
from networkx import connected_components, Graph
from toolz import itemmap, valmap, valfilter

from possibilities import Poss, PossibilityMonad
from games import logger
from games.game_def import (
    Game,
    GamePlayer,
    JointState,
    PlayerName,
    JointPureActions,
    RJ,
    RP,
    SR,
    U,
    X,
    Y,
)
from games.utils import fkeyfilter, iterate_dict_combinations, fvalmap
from games.create_joint_game_tree import create_game_graph
from games.performance import GetFactorizationPI
from games.solve.solution import solve_game2
from games.solve.solution_structures import (
    GameFactorization,
    GamePlayerPreprocessed,
    SolvedGameNode,
    UsedResources,
    SolverParams
)
from games.access import collapse_states, flatten_sets


M = Mapping


def get_game_factorization_no_collision_check(
    game: Game[X, U, Y, RP, RJ, SR],
    players_pre: Mapping[PlayerName, GamePlayerPreprocessed[X, U, Y, RP, RJ, SR]],
    fact_perf: Optional[GetFactorizationPI] = None
) -> GameFactorization[X]:
    """
    This functions returns the factorization for all jointly reachable states (goes through each state contained
    in the game graph).

    This function is not suited for more than 2 players if the resources of the game are considered (beta=0)
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
    Returns the dependency structure from the use of shared resources only of single player games.
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


def get_game_factorization_as_create_game_graph(
        game: Game[X, U, Y, RP, RJ, SR],
        solver_params: SolverParams,
        players_pre: Mapping[PlayerName, GamePlayerPreprocessed[X, U, Y, RP, RJ, SR]],
        fact_perf: Optional[GetFactorizationPI] = None
) -> GameFactorization[X]:
    """
    This functions returns the factorization for all jointly reachable states (goes through each state contained
    in the game graph).

    This function is not suited for more than 2 players if the resources of the game are considered (beta=0)

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
        single_js = frozendict({pname: state})
        return pname, known[pname][single_js].ur

    # iterate all combinations
    for jsf in recursive_reachable_state_iterator(game=game, dt=solver_params.dt):

        special = all(_.x == 0 for _ in jsf.values())

        resources_used = itemmap(get_ur, jsf)

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


def get_game_factorization_n_players_as_create_game_graph(
        game: Game[X, U, Y, RP, RJ, SR],
        solver_params: SolverParams,
        players_pre: Mapping[PlayerName, GamePlayerPreprocessed[X, U, Y, RP, RJ, SR]],
        fact_perf: Optional[GetFactorizationPI] = None
) -> GameFactorization[X]:
    """
    This functions returns the factorization for all jointly reachable states (goes through each state contained
    in the game graph).

    For more than 2 players when considering the resources of the game (beta=0) the subgames that influence each other
    are iteratively solved to check for influence between each other based on the future resources of those subgames

    :param game:
    :param solver_params:
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
        single_js = frozendict({pname: state})
        return pname, known[pname][single_js].ur

    solved_to_solution = {}
    # solved_to_solution.update(known)

    # iterate all combinations
    for jsf in recursive_reachable_state_iterator(game=game, dt=solver_params.dt):

        special = all(_.x == 0 for _ in jsf.values())

        resources_used = itemmap(get_ur, jsf)

        # start timer to collect the time for finding dependencies between players
        t1 = perf_counter()

        deps = find_dependencies_n_players_no_collision_check(
            game,
            solver_params,
            jsf,
            players_pre,
            ps,
            resources_used,
            solved_to_solution
        )

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


def find_dependencies_n_players_no_collision_check(
    game: Game[X, U, Y, RP, RJ, SR],
    solver_params: SolverParams,
    current_js: JointState,
    players_pre: Mapping[PlayerName, GamePlayerPreprocessed[X, U, Y, RP, RJ, SR]],
    ps: PossibilityMonad,
    resources_used: Mapping[PlayerName, UsedResources[X, U, Y, RP, RJ, SR]],
    states_to_solution: Dict[JointState, SolvedGameNode]
) -> Mapping[FSet[PlayerName], FSet[FSet[PlayerName]]]:
    """
    Returns the dependency structure from the use of shared resources.
    Returns the partitions of players that are independent.
    For more than 2 players when considering the resources of the game (beta=0) the subgames that influence each other
    are iteratively solved to check again for influence between each other.

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

    if hasattr(solver_params, "beta"):
        beta = solver_params.beta
    else:
        beta = 0

    players = set(resources_used)
    n = len(players)

    condensation_graph = list(connected_components(interaction_graph))
    nb_games = len(condensation_graph)

    if beta == math.inf or nb_games == n or nb_games == 1:
        """
        In this case the single player games do not have to be merged
        """
        result = {}
        for nplayers in range(2, n + 1):
            for players_subset in itertools.combinations(players, nplayers):
                G = interaction_graph.subgraph(players_subset)
                r = frozenset(map(frozenset, connected_components(G)))
                result[frozenset(players_subset)] = r
        return result

    else:
        """
        The games in the partitions of the condensation of the interaction graph have to be solved and the resources
        of those games have to be compared in order to be sure to not lead into a collision. 
        """
        condensation_graph_old = frozenset()
        condensation_graph = frozenset(map(frozenset, condensation_graph))  # make immutable

        while condensation_graph_old != condensation_graph:  # as long as still games were merged
            condensation_graph_old = condensation_graph
            interaction_graph = Graph()
            interaction_graph.add_nodes_from(condensation_graph_old)  # nodes are set of players (smaller games)

            # for each merged subgame the corresponding resources
            resources_games: Dict[FSet[PlayerName], UsedResources] = {}
            for players in condensation_graph_old:  # iterate through "games"

                smaller_game: Game[X, U, Y, RP, RJ, SR]
                # get the game for the current players only, i.e. merge the single player games
                smaller_game = get_smaller_game(game, players)

                # initial state of those subgame
                initials_smaller_game = fkeyfilter(lambda _: _ in players, current_js)

                # create the game graph for the merged sub game
                gg = create_game_graph(
                    game=smaller_game,
                    dt=solver_params.dt,
                    initials=frozenset({initials_smaller_game}),
                    gf=None
                )

                # solve the subgame
                gs = solve_game2(
                    game=smaller_game,
                    solver_params=solver_params,
                    gg=gg,
                    jss=frozenset({initials_smaller_game})
                )

                resources_games[players] = gs.states_to_solution[initials_smaller_game].ur

            max_instants = max(max(_.used) if _.used else 0 for _ in resources_games.values())
            for i in range(int(max_instants) + 1):
                i = D(i)

                def getused_game(items: Tuple[FSet[PlayerName], UsedResources]) -> Tuple[FSet[PlayerName], FSet[SR]]:
                    """
                    Get the shared resources of the games
                    """
                    ur: UsedResources
                    players_g, ur = items
                    used: Mapping[D, Poss[Mapping[PlayerName, FSet[SR]]]] = ur.used
                    if i not in used:
                        res = frozenset()
                    else:
                        at_i: Poss[Mapping[PlayerName, FSet[SR]]] = ur.used[i]
                        at_i_players: Poss[FSet[SR]]
                        # It could be that the player already finished for some actions (does not use any resources)
                        # and for some actions he didn't finish (uses resources) -> return default value empty set
                        res = frozenset()
                        for player_name in players_g:
                            at_i_player = ps.build(at_i, lambda _: _.get(player_name, frozenset()))
                            support_sets = flatten_sets(at_i_player.support())
                            res |= support_sets
                    return players_g, res

                used_at_i: Dict[FSet[PlayerName], FSet[SR]]
                used_at_i = itemmap(getused_game, resources_games)

                #iterate through the games and check if resources of the games intersect
                for players_g1, players_g2 in itertools.combinations(condensation_graph_old, 2):
                    intersection = used_at_i[players_g1] & used_at_i[players_g2]
                    intersects = len(intersection) > 0
                    if intersects:
                        interaction_graph.add_edge(players_g1, players_g2)

            # merge the games that are influencing each other
            _condensation_graph = frozenset(map(frozenset, connected_components(interaction_graph)))
            condensation_graph = frozenset(map(flatten_sets, _condensation_graph))

            if len(condensation_graph) == 1: # if only a single game is left break
                break

        players = set(resources_used)  # all the players present at current game node
        n = len(players)  # number of players present

        result = {}

        # iterate through all combinations of subsets of players
        for nplayers in range(2, n + 1):
            for players_subset in itertools.combinations(players, nplayers):
                # get the condensation graph only for the players left
                condensation_graph_subset = frozenset(map(lambda _: _ & frozenset(players_subset), condensation_graph))
                # remove all the empty partitions
                r = frozenset(filter(lambda _: _ != frozenset(), condensation_graph_subset))
                result[frozenset(players_subset)] = r
        return result


def get_smaller_game(
        game: Game[X, U, Y, RP, RJ, SR],
        players: FSet[PlayerName]
) -> Game[X, U, Y, RP, RJ, SR]:
    """ Returns the individual games (by removing all others players)"""

    game_players = fkeyfilter(lambda _: _ in players, game.players)
    g = replace(game, players=game_players)
    return g


def recursive_reachable_state_iterator(
    game: Game[X, U, Y, RP, RJ, SR],
    dt: D,
) -> Iterator[JointState]:

    reachable_states = set()
    initials = get_initial_states(game.players)
    for js in initials:
        yield from _recursive_reachable_state_iterator(game, reachable_states, js, dt)


def _recursive_reachable_state_iterator(
        game,
        reachable_states: Set[JointState],
        states: JointState,
        dt: D
) -> Iterator[JointState]:
    """
    """
    if states not in reachable_states:

        moves_to_state_everybody = _get_moves(game, states, dt)
        pure_outcomes: Dict[JointPureActions, Poss[Mapping[PlayerName, JointState]]] = {}
        ps = game.ps

        is_final = {}
        for player_name, player_state in states.items():
            _ = game.players[player_name]
            if _.personal_reward_structure.is_personal_final_state(player_state):
                f = _.personal_reward_structure.personal_final_reward(player_state)
                is_final[player_name] = f

        who_exits = frozenset(game.joint_reward.is_joint_final_state(states))
        joint_final = who_exits

        players_exiting = set(who_exits) | set(is_final)

        # Consider only the moves of who remains
        not_exiting = lambda pn: pn not in players_exiting
        moves_to_state_remaining = fkeyfilter(not_exiting, moves_to_state_everybody)
        movesets_for_remaining = fvalmap(frozenset, moves_to_state_remaining)

        for joint_pure_action in iterate_dict_combinations(moves_to_state_remaining):

            pure_action: JointPureActions
            pure_action = fkeyfilter(lambda action: action is not None, joint_pure_action)

            if not pure_action:
                continue

            def f(item: Tuple[PlayerName, U]) -> Tuple[PlayerName, Poss[X]]:
                pn, choice = item
                # fixme ps.lift_many(moves_to_state_remaining[pn][choice])?
                return pn, moves_to_state_remaining[pn][choice]

            selected: Dict[PlayerName, Poss[X]]
            selected = itemmap(f, pure_action)

            def f(a: Mapping[PlayerName, U]) -> JointState:
                return fkeyfilter(not_exiting, a)

            outcomes: Poss[JointState] = ps.build_multiple(selected, f)

            def r(js0: JointState) -> Mapping[PlayerName, JointState]:
                x = {k_: js0 for k_ in states}
                return fkeyfilter(not_exiting, x)

            poutcomes: Poss[Mapping[PlayerName, JointState]] = ps.build(outcomes, r)
            pure_outcomes[pure_action] = poutcomes

            for p in poutcomes.support():
                for _, js_ in p.items():
                    yield from _recursive_reachable_state_iterator(game, reachable_states, js_, dt)
        reachable_states.add(states)
        yield states


def _get_moves(
    game: Game[X, U, Y, RP, RJ, SR], js: JointState, dt: D
) -> Mapping[PlayerName, Mapping[U, Poss[X]]]:
    """ Returns the possible moves. """
    res = {}
    state: X
    ps = game.ps
    
    for player_name, state in js.items():
        player = game.players[player_name]
        # is it a final state?
        is_final = player.personal_reward_structure.is_personal_final_state(state) if state else True

        if state is None or is_final:
            succ = {None: ps.unit(None)}
        else:
            succ = player.dynamics.successors(state, dt)
        res[player_name] = succ
    return res


def get_initial_states(game_players: M[PlayerName, GamePlayer[X, U, Y, RP, RJ, SR]]) -> FSet[JointState]:
    initials_dict: Dict[PlayerName, List[X]] = {}
    for player_name, game_pl_pre in game_players.items():
        initials_support = game_pl_pre.initial.support()
        initials_dict[player_name] = []
        for _ini in initials_support:
            initials_dict[player_name].append(_ini)

    initials = frozenset((iterate_dict_combinations(initials_dict)))
    return initials


def reachable_states_iterator(
    game: Game[X, U, Y, RP, RJ, SR],
    dt: D
) -> Iterator[JointState]:

    initials = get_initial_states(game.players)

    stack: List[JointState] = []
    reachable_states = set()
    for js_initial in initials:
        stack.append(js_initial)
        while stack:
            js = stack.pop()

            def comp_items(_):
                return js.items() <= _.items()

            if any(map(comp_items, reachable_states)):
                continue

            reachable_states.add(js)
            yield js
            players_moves_to_pstates: Mapping[PlayerName, Mapping[U, Poss[X]]]
            players_moves_to_pstates = _get_moves(game=game, js=js, dt=dt)

            def get_set_states(mtps: Mapping[U, Poss[X]]) -> FSet[X]:
                states = itertools.chain(
                    *[filter(lambda _: _ is not None, psj.support()) for psj in mtps.values()]
                )
                return frozenset(states)

            _player_to_states = valmap(get_set_states, players_moves_to_pstates)
            player_to_states = valfilter(lambda _: len(_) > 0, _player_to_states)
            for js_new in iterate_dict_combinations(player_to_states):
                if js_new:
                    stack.append(js_new)
