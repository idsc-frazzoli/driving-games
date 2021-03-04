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
    Iterator,
    List
)

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
from games.performance import GetFactorizationPI
from games.solve.solution_structures import (
    GameFactorization,
    GamePlayerPreprocessed,
    SolvedGameNode,
    UsedResources,
)
from games.access import collapse_states, flatten_sets


M = Mapping


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


def get_game_factorization_as_create_game_graph(
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
        single_js = frozendict({pname: state})
        return pname, known[pname][single_js].ur

    # iterate all combinations
    for jsf in recursive_reachable_state_iterator(game=game):

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


def recursive_reachable_state_iterator(
    game: Game[X, U, Y, RP, RJ, SR],
    dt: D = D(1),
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
    dt: D = D(1)  # fixme pass it to the function for other timesteps
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
