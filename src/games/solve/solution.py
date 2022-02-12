from collections import defaultdict
from decimal import Decimal as D
from time import perf_counter
from typing import AbstractSet, Dict, FrozenSet as FSet, Mapping as M, Optional

from cytoolz import valmap
from frozendict import frozendict
from networkx import simple_cycles
from zuper_commons.types import ZValueError

from dg_commons import X, U, Y, RP, RJ, PlayerName, fd
from games import logger
from games.agent_from_policy import AgentFromPolicy
from games.checks import check_joint_state
from games.create_joint_game_tree import create_game_graph
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
from games.solve.solution_ghost import get_ghost_tree
from games.solve.solution_structures import (
    GameGraph,
    GameNode,
    GamePreprocessed,
    GameSolution,
    Solutions,
    SolutionsPlayer,
    SolvedGameNode,
    SolverParams,
    SolvingContext,
    UsedResources,
    ValueAndActions,
)
from games.solve.solution_utils import get_outcome_preferences_for_players, add_incremental_cost_player, fd_r
from games.solve.solve_equilibria import solve_equilibria, solve_final_for_everyone
from possibilities import Poss

__all__ = ["solve_main"]

TOC = perf_counter()


def solve_main(
    gp: GamePreprocessed[X, U, Y, RP, RJ, SR], perf_stats: Optional[PerformanceStatistics] = None
) -> Solutions[X, U, Y, RP, RJ, SR]:
    """
    Documentation todo

    :param gp:
    :return:
    """
    G = gp.game_graph
    dt = gp.solver_params.dt
    # find initial states
    # noinspection PyCallingNonCallable
    initials = list((node for node, degree in G.in_degree() if degree == 0))

    logger.info(initials=initials)
    assert len(initials) == 1
    initial = initials[0]

    # noinspection PyCallingNonCallable
    finals = list(node for node, degree in G.out_degree() if degree == 0)
    logger.info(finals=len(finals))

    cycles = list(simple_cycles(G))
    if cycles:
        msg = "Did not expect cycles in the graph"
        raise ZValueError(msg, cycles=cycles)

    # We will fill this with some simulations of different policies
    sims: Dict[str, Simulation] = {}

    # Use game factorization only if the option is set
    if gp.game_factorization and gp.solver_params.use_factorization:
        gf = gp.game_factorization
    else:
        gf = None

    gg = create_game_graph(gp.game, gp.solver_params.dt, {initial}, gf=gf)

    game_tree = gg.state2node[initial]
    solutions_players: Dict[PlayerName, SolutionsPlayer[X, U, Y, RP, RJ, SR]] = {}
    initial_state = game_tree.states
    # solve sequential games equilibria #todo not defined for n>2
    # sims = solve_sequential_games(gp=gp, gg=gg, initial_state=initial_state, sims=sims)

    # solve simultaneous play (Nash equilibria)
    logger.info("solving game tree")
    game_solution = solve_game(game=gp.game, gg=gg, solver_params=gp.solver_params, jss=initials)
    controllers0 = {}
    for player_name, pp in gp.players_pre.items():
        policy = game_solution.policies[player_name]
        controllers0[player_name] = AgentFromPolicy(gp.game.ps, policy)

    logger.info(
        f"Value of joint solution",
        game_value=game_solution.states_to_solution[initial_state].va.game_value,
        # policy=solution_ghost.policies,
    )
    for seed in range(gp.solver_params.n_simulations):
        sim_joint = simulate1(
            gp.game,
            policies=controllers0,
            initial_states=game_tree.states,
            dt=gp.solver_params.dt,
            seed=seed,
        )
        sims[f"joint-{seed}"] = sim_joint

    return Solutions(
        game_solution=game_solution,
        game_tree=game_tree,
        solutions_players=solutions_players,
        sims=sims,
    )
    # logger.info(game_tree=game_tree)


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


def solve_game(
    *,
    game: Game[X, U, Y, RP, RJ, SR],
    solver_params: SolverParams,
    gg: GameGraph[X, U, Y, RP, RJ, SR],
    jss: AbstractSet[JointState],
) -> GameSolution[X, U, Y, RP, RJ, SR]:
    """
    Computes the solution of the game rooted in `jss` and extract the policy for each player, for each game node

    :param game:
    :param solver_params:
    :param gg:
    :param jss:
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
    )
    for js0 in jss:
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
        initials=frozenset(jss),
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
        va = solve_final_for_everyone(sc, gn)
    else:
        va = solve_equilibria(sc, gn, solved)

    ur = get_used_resources(sc, va=va, gn=gn, solved_to_node=solved_to_node)

    try:
        ret = SolvedGameNode(states=js, solved=fd(solved_to_node), va=va, ur=ur)
    except Exception as e:
        raise ZValueError(game_node=gn) from e
    sc.cache[js] = ret
    sc.processing.remove(js)

    n = len(sc.cache)
    if n % 50 == 0:
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
) -> UsedResources[X, U, Y, RP, RJ, SR]:
    """Computes the future resources of a solved sub game (game node 'gn')"""
    ps = sc.game.ps
    usage_current = ps.unit(gn.resources)
    # logger.info(va=va)
    if va.mixed_actions:  # i.e. it's not a terminal node
        next_states: Poss[M[PlayerName, JointState]]
        next_states = ps.join(ps.build_multiple(va.mixed_actions, solved_to_node.__getitem__))

        usages: Dict[D, Poss[M[PlayerName, FSet[SR]]]]
        usages = {D(0): usage_current}
        # Π = 1
        i = D(0)
        while True:  # todo: use the range that's needed
            default = ps.unit(frozendict())

            def get_data(x: M[PlayerName, JointState]) -> Poss[M[PlayerName, FSet[SR]]]:
                used_by_players: Dict[PlayerName, Poss[FSet[SR]]] = {}
                for pname in va.mixed_actions:

                    def get_its(y: M[PlayerName, FSet[SR]]) -> FSet[SR]:
                        return y.get(pname, frozenset())

                    st = x[pname]
                    gn_ = sc.cache[st]
                    ui = gn_.ur.used.get(i, default)
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

            at_d = ps.build(next_states, get_data)
            f = ps.join(at_d)
            if f.support() != {frozendict()}:
                usages[i + 1] = f
            else:
                break
            # fixme here you need dt? or the stage index?
            i += 1

        logger.info(usages=usages)  #
        ur = UsedResources(fd(usages))
    else:
        usages_ = fd({D(0): usage_current})
        ur = UsedResources(usages_)
    return ur
