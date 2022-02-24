from collections import defaultdict
from decimal import Decimal as D
from functools import partial
from time import perf_counter
from typing import Dict, FrozenSet as FSet, List, Mapping, Mapping as M, Tuple, AbstractSet

from cytoolz import valmap
from frozendict import frozendict
from zuper_commons.types import ZValueError

from dg_commons import fd, iterate_dict_combinations, PlayerName, RJ, RP, U, X, Y
from games import logger
from games.agent_from_policy import AgentFromPolicy
from games.checks import check_joint_state
from games.game_def import (
    Combined,
    Game,
    JointPureActions,
    JointState,
    SR,
    UncertainCombined,
)
from games.performance import PerformanceStatistics
from games.simulate import simulate1, Simulation
from possibilities import Poss
from .solution_ghost import get_ghost_tree
from .solution_structures import (
    GameGraph,
    GameNode,
    GamePreprocessed,
    GameSolution,
    Solutions,
    SolvedGameNode,
    SolverParams,
    SolvingContext,
    UsedResources,
    ValueAndActions,
    ResourcesType,
    SolutionsPlayer,
)
from .solution_utils import add_incremental_cost_player, fd_r, get_outcome_preferences_for_players
from .solve_equilibria import solve_equilibria, solve_final_for_everyone
from ..create_joint_game_graph import create_game_graph
from ..factorization import collapse_states
from ..game_graph_to_nx import build_networkx_from_game_graph, compute_graph_layout

__all__ = ["solve_main"]


def solve_main(
    gp: GamePreprocessed[X, U, Y, RP, RJ, SR],
    perf_stats: PerformanceStatistics,
) -> Solutions[X, U, Y, RP, RJ, SR]:
    """
    Documentation todo
    :param gp:
    :param perf_stats: Object to collect performance insights
    :return:
    """

    init_states_players: Mapping[PlayerName, X] = valmap(lambda x: x.initial.support(), gp.game.players)
    init_states: List[JointState] = [js for js in iterate_dict_combinations(init_states_players)]
    assert len(init_states) == 1, init_states
    no_fact_initial = init_states[0]
    logger.info(initial=no_fact_initial)

    tic = perf_counter()
    # Create the game graph
    gg = create_game_graph(
        game=gp.game,
        dt=gp.solver_params.dt,
        initials={no_fact_initial},
        players_pre=gp.players_pre,
        fact_algo=gp.solver_params.factorization_algorithm,
        compute_res=False,
        max_depth=gp.solver_params.max_depth,
    )
    toc = perf_counter()
    perf_stats.build_joint_game_tree = toc - tic
    perf_stats.joint_game_tree_nodes = len(gg.state2node)

    if gp.solver_params.extra:
        limit_nodes = 5000
        if len(gg.state2node) > limit_nodes:
            logger.info(
                f"Attempting to create networkx graph with more than {limit_nodes} nodes.\n"
                f'If stuck retry with "noextra" option for the solver.'
            )
        game_graph_nx = build_networkx_from_game_graph(gg)
        compute_graph_layout(game_graph_nx, iterations=1)
    else:
        game_graph_nx = None

    # solve sequential games equilibria # todo not defined for n>2
    # sims = solve_sequential_games(gp=gp, gg=gg, initial_state=initial_state, sims=sims)
    # get the game graphs

    # why is this not used?!
    solutions_players: Dict[PlayerName, SolutionsPlayer[X, U, Y, RP, RJ, SR]] = {}

    # solve simultaneous play (Nash equilibria)
    logger.info("Solving joint game tree")
    global TOC
    TOC = perf_counter()
    tic = perf_counter()
    factorize = gp.solver_params.factorization_algorithm.factorize
    fact_jss: FSet[JointState] = frozenset(
        factorize(no_fact_initial, known=valmap(collapse_states, gp.players_pre), ps=gp.game.ps).values()
    )

    game_solution = solve_game(
        game=gp.game, gg=gg, initials=fact_jss, solver_params=gp.solver_params, compute_res=False
    )
    toc = perf_counter()
    perf_stats.solve_joint_game_graph = toc - tic

    controllers0 = {}
    for player_name in gp.game.players:
        policy = game_solution.policies[player_name]
        controllers0[player_name] = AgentFromPolicy(gp.game.ps, policy, player_name)

    logger.info(
        f"Value of initial states joint solution",
        game_values=[game_solution.states_to_solution[js_].va.game_value for js_ in fact_jss],
        # policy=solution_ghost.policies,
    )

    # We will fill this with some simulations of different policies
    sims: Dict[str, Simulation] = {}
    for seed in range(gp.solver_params.n_simulations):
        sim_joint = simulate1(
            gp.game,
            policies=controllers0,
            initial_states=no_fact_initial,
            dt=gp.solver_params.dt,
            seed=seed,
            max_stages=gp.solver_params.max_depth,
        )
        sims[f"joint-{seed}"] = sim_joint

    return Solutions(
        game_solution=game_solution,
        game_graph=gg,
        solutions_players=solutions_players,
        sims=sims,
        game_graph_nx=game_graph_nx,
    )


def solve_game(
    *,
    game: Game[X, U, Y, RP, RJ, SR],
    solver_params: SolverParams,
    gg: GameGraph[X, U, Y, RP, RJ, SR],
    initials: AbstractSet[JointState],
    compute_res: bool = True,
) -> GameSolution[X, U, Y, RP, RJ, SR]:
    """
    Computes the solution of the game rooted in `jss` and extract the policy for each player, for each game
    node

    :param game:
    :param solver_params:
    :param gg: The Game Graph
    :param initials: The initials states, must be already factorized!
    :param compute_res: weather or not to collect the used resources by the game nodes
    :return:
    """
    outcome_preferences = get_outcome_preferences_for_players(game)
    states_to_solution: Dict[JointState, SolvedGameNode] = {}
    sc = SolvingContext(
        game=game,
        outcome_preferences=outcome_preferences,
        gg=gg,
        cache=states_to_solution,
        processing=set(),
        solver_params=solver_params,
        compute_res=compute_res,
    )

    for js0 in initials:
        _solve_game(sc, js0)

    policies: Dict[PlayerName, Dict[X, Dict[Poss[JointState], Poss[U]]]]
    ps = game.ps
    policies = defaultdict(lambda: defaultdict(dict))
    for state, solved_gnode in states_to_solution.items():
        for player_name, player_state in state.items():
            if player_name in solved_gnode.va.mixed_actions:
                policy_for_this_state = policies[player_name][player_state]
                other_states = fd({k: v for k, v in state.items() if k != player_name})
                iset = ps.unit(other_states)
                policy_for_this_state[iset] = solved_gnode.va.mixed_actions[player_name]

    policies2 = fd({k: fd_r(v) for k, v in policies.items()})

    return GameSolution(
        initials=gg.initials,
        policies=policies2,
        states_to_solution=fd(states_to_solution),
    )


def _solve_game(
    sc: SolvingContext[X, U, Y, RP, RJ, SR],
    js: JointState,
) -> SolvedGameNode[X, U, Y, RP, RJ, SR]:
    """
    Actual recursive function that solves the game nodes via backward induction

    :param sc: the solving context that is modified in place
    :param js: the current joint state
    :return: a solved game node
    """
    check_joint_state(js)
    if not js:
        raise ZValueError(js=js)
    if js in sc.cache:
        return sc.cache[js]
    if js in sc.processing:
        msg = "Loop found"
        raise ZValueError(msg, states=js)
    gn: GameNode[X, U, Y, RP, RJ, SR] = sc.gg.state2node[js]
    sc.processing.add(js)

    ps = sc.game.ps
    # what happens for each pure action?
    pure_actions: JointPureActions
    solved: Dict[JointPureActions, M[PlayerName, UncertainCombined]] = {}
    solved_to_node: Dict[JointPureActions, Poss[M[PlayerName, JointState]]]
    solved_to_node = {}

    for pure_actions in gn.transitions:
        # These are the solved nodes; for each, we find the solutions (recursive step here)
        # def u(a: M[PlayerName, JointState]) -> M[PlayerName, SolvedGameNode[X, U, U, RP, RJ, SR]]:
        def u(a: M[PlayerName, JointState]) -> M[PlayerName, JointState]:
            return fd(valmap(lambda _: _solve_game(sc, _).states, a))

        # Note that each player can go in a different joint state.
        next_nodes: Poss[M[PlayerName, JointState]] = gn.transitions[pure_actions]
        solved_to_node[pure_actions] = ps.build(next_nodes, u)
        players_dist: Dict[PlayerName, UncertainCombined] = {}
        # Incremental costs incurred if choosing this action
        inc: Poss[M[PlayerName, Combined[RJ, RP]]]
        inc = gn.incremental[pure_actions]

        for player_name in pure_actions:

            def v(m: M[PlayerName, JointState]) -> UncertainCombined:
                gn2: SolvedGameNode[X, U, U, RP, RJ, SR] = sc.cache[m[player_name]]
                if player_name not in gn2.va.game_value:
                    raise ZValueError(player_name=player_name, gn2=gn2, stn=stn)
                return gn2.va.game_value[player_name]

            stn = solved_to_node[pure_actions]
            # logger.info(stn=stn)
            player_dist: UncertainCombined = ps.join(ps.build(stn, v))

            def f(_: Combined) -> UncertainCombined:
                return add_incremental_cost_player(
                    game=sc.game,
                    player_name=player_name,
                    incremental=inc,
                    cur=_,
                )

            # logger.info(player_dist=player_dist)
            players_dist[player_name] = ps.join(ps.build(player_dist, f))
        # logger.info(players_dist=players_dist)
        solved[pure_actions] = fd(players_dist)

    va: ValueAndActions[U, RP, RJ]
    if set(gn.states) == set(gn.personal_final_reward) | set(gn.joint_final_rewards):
        # All the actives finish
        # todo for supporting max depth we land here if we reach max depth?
        va = solve_final_for_everyone(sc, gn)
    else:
        va = solve_equilibria(sc, gn, fd(solved))

    optr, reachr = None, None
    if sc.compute_res:
        optr, reachr = get_used_resources(sc, va=va, gn=gn, solved_to_node=solved_to_node)

    try:
        ret = SolvedGameNode(states=js, solved=fd(solved_to_node), va=va, optimal_res=optr, reachable_res=reachr)
    except Exception as e:
        raise ZValueError(game_node=gn) from e
    sc.cache[js] = ret
    sc.processing.remove(js)

    n = len(sc.cache)
    if n % 1e4 == 0:
        global TOC
        logger.info(
            js=js,
            states=gn.states,
            value=va.game_value,
            processing=len(sc.processing),
            solved=len(sc.cache),
            delta_time=perf_counter() - TOC,
        )
        TOC = perf_counter()
        # logger.info(f"nsolved: {n}")  # , game_value=va.game_value)
    return ret


def get_used_resources(
    sc: SolvingContext,
    va: ValueAndActions,
    gn: GameNode,
    solved_to_node: Dict[JointPureActions, Poss[M[PlayerName, JointState]]],
) -> Tuple[UsedResources, UsedResources]:
    """
    Computes the future resources of a solved sub game (game node 'gn')
    Assumes future nodes are already solved and in sc.cache
    :returns: (optimal_res, reachable_res)
    """
    ps = sc.game.ps
    usage_current = ps.unit(gn.resources)
    i = D(0)
    opt_res: Dict[D, Poss[M[PlayerName, FSet[SR]]]]
    opt_res = {i: usage_current}
    reach_res = {i: usage_current}

    # logger.info(va=va)
    if va.mixed_actions:  # i.e. it's not a terminal node
        opt_next_states: Poss[M[PlayerName, JointState]]
        opt_next_states = ps.join(ps.build_multiple(va.mixed_actions, solved_to_node.__getitem__))
        reach_next_states: Poss[M[PlayerName, JointState]]

        def f(a: Mapping[JointPureActions, Mapping[PlayerName, JointState]]) -> Poss[Mapping[PlayerName, JointState]]:
            return ps.lift_many(a.values())

        reach_next_states = ps.join(ps.build_multiple(gn.transitions, f))

        while True:
            default = ps.unit(frozendict())

            def get_data(x: M[PlayerName, JointState], res_type: ResourcesType) -> Poss[M[PlayerName, FSet[SR]]]:
                used_by_players: Dict[PlayerName, Poss[FSet[SR]]] = {}
                for pname in va.mixed_actions:

                    def get_its(y: M[PlayerName, FSet[SR]]) -> FSet[SR]:
                        return y.get(pname, frozenset())

                    st = x[pname]
                    gn_ = sc.cache[st]  # recursive part
                    if res_type == ResourcesType.OPTIMAL:
                        ui = gn_.optimal_res.used.get(i, default)
                    elif res_type == ResourcesType.REACHABLE:
                        ui = gn_.reachable_res.used.get(i, default)
                    else:
                        raise ZValueError("Unknown resource type", res_type=res_type)
                    used_at_i_by_player: Poss[FSet[SR]] = ps.build(ui, get_its)
                    used_by_players[pname] = used_at_i_by_player

                def remove_empty(_: M[PlayerName, FSet[SR]]) -> M[PlayerName, FSet[SR]]:
                    notempty = {}
                    for k, sr_used in _.items():
                        if sr_used:
                            notempty[k] = sr_used
                    return fd(notempty)

                res: Poss[M[PlayerName, FSet[SR]]]
                res = ps.build_multiple(used_by_players, remove_empty)
                return res

            get_data_optimal = partial(get_data, res_type=ResourcesType.OPTIMAL)
            get_data_reach = partial(get_data, res_type=ResourcesType.REACHABLE)

            opt_at_d = ps.join(ps.build(opt_next_states, get_data_optimal))
            reac_at_d = ps.join(ps.build(reach_next_states, get_data_reach))

            if opt_at_d.support() != {frozendict()}:
                opt_res[i + 1] = opt_at_d
            if reac_at_d.support() != {frozendict()}:
                reach_res[i + 1] = reac_at_d
            else:
                # opt resources can be done while reachable not. But not the converse...
                if opt_at_d.support() != reac_at_d.support():
                    raise ZValueError(
                        "Optimal and reachable resources are not equal", opt_at_d=opt_at_d, reac_at_d=reac_at_d
                    )
                break
            # The stage index
            i += 1
        # logger.info(usages=usages)
    opt_ur = UsedResources(fd(opt_res))
    reach_ur = UsedResources(fd(reach_res))
    return opt_ur, reach_ur


# fixme XXXXX beyond here functions are deprecated


def solve_sequential_games(
    gp: GamePreprocessed, gg: GameGraph, initial_state: JointState, sims: Dict[str, Simulation]
) -> Dict[str, Simulation]:
    """
    #todo these are currently not defined for n>2 players
    :param gp:
    :param gg:
    :param initial_state:
    :param sims:
    :return:
    """
    for player_name, pp in gp.players_pre.items():
        # use other solutions
        controllers_others = {}
        for p2 in gp.players_pre:
            if p2 == player_name:
                continue
            x_p2 = initial_state[p2]
            policy = gp.players_pre[p2].gs.policies[p2]
            controllers_others[p2] = AgentFromPolicy(gp.game.ps, policy)

        ghost_game_graph = get_ghost_tree(gp.game, player_name, gg, controllers_others)
        logger.info("The game graph has dimension", nnodes=len(ghost_game_graph.state2node))
        solution_ghost = solve_game(
            game=gp.game,
            gg=ghost_game_graph,
            solver_params=gp.solver_params,
            jss={initial_state},
        )
        msg = f"Sequential solution when {player_name} plays last"
        game_values = solution_ghost.states_to_solution[initial_state].va.game_value
        logger.info(msg, game_values=game_values)

        controllers = dict(controllers_others)
        controllers[player_name] = AgentFromPolicy(gp.game.ps, solution_ghost.policies[player_name])
        sim_ = simulate1(
            gp.game,
            policies=controllers,
            initial_states=initial_state,
            dt=gp.solver_params.dt,
            seed=0,
        )
        sims[f"{player_name}-follows"] = sim_
    return sims
