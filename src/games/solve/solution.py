from collections import defaultdict
from decimal import Decimal as D
from time import perf_counter
from typing import (
    AbstractSet,
    Dict,
    FrozenSet as FSet,
    Mapping as M,
    List
)

from frozendict import frozendict
from networkx import simple_cycles
from toolz import valmap

from possibilities import Poss
from zuper_commons.types import ZValueError
from games import logger
from games.agent_from_policy import AgentFromPolicy
from games.create_joint_game_tree import create_game_graph
from games.utils import iterate_dict_combinations
from games.game_def import (
    GamePlayer,
    check_joint_state,
    Combined,
    Game,
    JointPureActions,
    JointState,
    P,
    PlayerName,
    RJ,
    RP,
    SR,
    U,
    UncertainCombined,
    X,
    Y,
)
from games.simulate import simulate1, Simulation
from .solution_ghost import get_ghost_tree
from .solution_utils import get_outcome_preferences_for_players, add_incremental_cost_single, fr
from .solve_equilibria_ import solve_equilibria, get_game_values_final
from .solution_structures import (
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

__all__ = ["solve1"]

TOC = perf_counter()


def solve1(gp: GamePreprocessed[X, U, Y, RP, RJ, SR]) -> Solutions[X, U, Y, RP, RJ, SR]:
    """
    Documentation todo

    :param gp:
    :return:
    """
    G = gp.game_graph
    dt = gp.solver_params.dt
    # find initial states
    # noinspection PyCallingNonCallable

    initials = get_initial_states(game_prepro=gp)

    # initials_old = list((node for node, degree in G.in_degree() if degree == 0))

    logger.info(msg="Initial states of all players", initials=initials)
    assert len(initials) == 1
    initial = initials[0]

    # noinspection PyCallingNonCallable
    finals = list(node for node, degree in G.out_degree() if degree == 0)
    logger.info(msg="Final states of the first two players", finals=len(finals))

    cycles = list(simple_cycles(G))
    if cycles:
        msg = "Did not expect cycles in the graph"
        raise ZValueError(msg, cycles=cycles)

    # We will fill this with some simulations of different policies
    sims: Dict[str, Simulation] = {}

    logger.info("creating game tree")

    # Use game factorization only if the option is set
    if gp.game_factorization and gp.solver_params.use_factorization:
        gf = gp.game_factorization
    else:
        gf = None

    gg = create_game_graph(gp.game, gp.solver_params.dt, {initial}, gf=gf)

    game_tree = gg.state2node[initial]
    solutions_players: Dict[PlayerName, SolutionsPlayer[X, U, Y, RP, RJ, SR]] = {}
    initial_state = game_tree.states
    # solve sequential games equilibria
    sims = solve_sequential_games(gp=gp, gg=gg, initial_state=initial_state, sims=sims)
    # solve simultaneous play (Nash equilibria)
    logger.info("solving game tree")
    game_solution = solve_game2(game=gp.game, gg=gg, solver_params=gp.solver_params, jss=initials)
    controllers0 = {}
    for player_name, pp in gp.players_pre.items():
        policy = game_solution.policies[player_name]
        controllers0[player_name] = AgentFromPolicy(gp.game.ps, policy)

    logger.info(
        f"Value of joint solution",
        game_value=game_solution.states_to_solution[initial_state].va.game_value,
        # policy=solution_ghost.policies,
    )
    for seed in range(5):
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

def get_initial_states(game_prepro: GamePreprocessed) -> List[JointState]:
    game_players: M[PlayerName, GamePlayer[X, U, Y, RP, RJ, SR]]
    game_players = game_prepro.game.players

    initials_dict: Dict[PlayerName, List[X]]  = {}
    for player_name, game_pl_pre in game_players.items():
        initials_support = game_pl_pre.initial.support()
        initials_dict[player_name] = []
        for _ini in initials_support:
            initials_dict[player_name].append(_ini)

    initials = list(iterate_dict_combinations(initials_dict))
    return initials


def solve_sequential_games(
    gp: GamePreprocessed, gg: GameGraph, initial_state: JointState, sims: Dict[str, Simulation]
) -> Dict[str, Simulation]:
    """#todo

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
        solution_ghost = solve_game2(
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


def solve_game2(
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
        check_joint_state(js0)
        _solve_game(sc, js0)

    policies: Dict[PlayerName, Dict[X, Dict[Poss[JointState], Poss[U]]]]
    ps = game.ps
    policies = defaultdict(lambda: defaultdict(dict))
    for state, solved_gnode in states_to_solution.items():
        for player_name, player_state in state.items():
            if player_name in solved_gnode.va.mixed_actions:
                policy_for_this_state = policies[player_name][player_state]
                other_states = frozendict({k: v for k, v in state.items() if k != player_name})
                iset = ps.unit(other_states)
                policy_for_this_state[iset] = solved_gnode.va.mixed_actions[player_name]

    policies2 = frozendict({k: fr(v) for k, v in policies.items()})

    return GameSolution(
        initials=frozenset(jss),
        policies=policies2,
        states_to_solution=frozendict(states_to_solution),
    )


def _solve_game(
    sc: SolvingContext[X, U, Y, RP, RJ, SR],
    js: JointState,
) -> SolvedGameNode[X, U, Y, RP, RJ, SR]:
    """
    Actual recursive function that solves the game nodes with backward induction

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

    for pure_actions in gn.outcomes:
        # Incremental costs incurred if choosing this action
        inc: Dict[PlayerName, RP]
        inc = {p: gn.incremental[p][u] for p, u in pure_actions.items()}
        # if we choose these actions, then these are the game nodes we could go in.
        # Note that each player can go in a different joint state.
        next_nodes: Poss[M[PlayerName, JointState]] = gn.outcomes[pure_actions]

        # These are the solved nodes; for each, we find the solutions (recursive step here)
        # def u(a: M[PlayerName, JointState]) -> M[PlayerName, SolvedGameNode[X, U, U, RP, RJ, SR]]:
        def u(a: M[PlayerName, JointState]) -> M[PlayerName, JointState]:
            return frozendict(valmap(lambda _: _solve_game(sc, _).states, a))

        solved_to_node[pure_actions] = ps.build(next_nodes, u)

        players_dist: Dict[PlayerName, UncertainCombined] = {}
        for player_name in pure_actions:

            def v(m: M[PlayerName, JointState]) -> UncertainCombined:
                gn2: SolvedGameNode[X, U, U, RP, RJ, SR] = sc.cache[m[player_name]]
                if not player_name in gn2.va.game_value:
                    raise ZValueError(player_name=player_name, gn2=gn2, stn=stn)
                return gn2.va.game_value[player_name]

            stn = solved_to_node[pure_actions]
            # logger.info(stn=stn)
            player_dist: UncertainCombined = ps.join(ps.build(stn, v))

            def f(_: Combined) -> Combined:
                return add_incremental_cost_single(
                    game=sc.game,
                    player_name=player_name,
                    incremental_for_player=inc,
                    cur=_,
                )

            # logger.info(player_dist=player_dist)
            players_dist[player_name] = ps.build(player_dist, f)

        # logger.info(players_dist=players_dist)
        solved[pure_actions] = frozendict(players_dist)

    va: ValueAndActions[U, RP, RJ]

    if set(gn.states) == set(gn.joint_final_rewards):  # final costs:
        # All finish jointly
        va = solve_final_joint(sc, gn)

    elif set(gn.states) == set(gn.is_final):
        # All the actives finish independently
        va = solve_final_personal_both(sc, gn)

    elif set(gn.states) == (set(gn.joint_final_rewards) | set(gn.is_final)):
        # all finish mixed
        va = solve_final_mixed(sc, gn)

    else:
        # there are still active players
        va = solve_equilibria(sc, gn, solved)

    ur: UsedResources[X, U, Y, RP, RJ, SR]
    usage_current = ps.unit(gn.resources)
    # logger.info(va=va)
    if va.mixed_actions:
        next_states: Poss[M[PlayerName, SolvedGameNode[X, U, U, RP, RJ, SR]]]
        # next_states: Poss[M[PlayerName, Poss[M[PlayerName, JointState]]]]
        next_states = ps.join(ps.build_multiple(va.mixed_actions, solved_to_node.__getitem__))

        usages: Dict[D, Poss[M[PlayerName, FSet[SR]]]]
        usages = {D(0): usage_current}
        # Π = 1

        for i in map(D, range(10)):  # todo: use the range that's needed
            default = ps.unit(frozendict())

            def get_data(x: M[PlayerName, JointState]) -> Poss[M[PlayerName, FSet[SR]]]:
                used_by_players: Dict[PlayerName, Poss[FSet[SR]]] = {}
                for pname in va.mixed_actions:

                    def get_its(y: M[PlayerName, FSet[SR]]) -> FSet[SR]:
                        return y.get(pname, frozenset())

                    st = x[player_name]
                    gn_ = sc.cache[st]
                    ui = gn_.ur.used.get(i, default)
                    used_at_i_by_player: Poss[FSet[SR]] = ps.build(ui, get_its)
                    used_by_players[pname] = used_at_i_by_player

                def remove_empty(_: M[PlayerName, FSet[SR]]) -> M[PlayerName, FSet[SR]]:
                    notempty = {}
                    for k, sr_used in _.items():
                        if sr_used:
                            notempty[k] = sr_used
                    return frozendict(notempty)

                res: Poss[M[PlayerName, FSet[SR]]]
                res = ps.build_multiple(used_by_players, remove_empty)
                return res

            at_d = ps.build(next_states, get_data)
            f = ps.join(at_d)
            if f.support() != {frozendict()}:
                usages[i + 1] = f

        # logger.info(next_resources=next_resources, usages=usages)
        ur = UsedResources(frozendict(usages))
    else:
        usages_ = frozendict({D(0): usage_current})
        ur = UsedResources(usages_)
    try:
        ret = SolvedGameNode(states=js, solved=frozendict(solved_to_node), va=va, ur=ur)
    except Exception as e:
        raise ZValueError(game_node=gn) from e
    sc.cache[js] = ret
    sc.processing.remove(js)

    n = len(sc.cache)
    if n % 30 == 0:
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

def solve_final_joint(
    sc: SolvingContext[X, U, Y, RP, RJ, SR], gn: GameNode[X, U, Y, RP, RJ, SR]
) -> ValueAndActions[U, RP, RJ]:
    """
    Solves a node which is a joint final node
    """
    game_value: Dict[PlayerName, UncertainCombined] = {}

    for player_name, joint in gn.joint_final_rewards.items():
        personal = sc.game.players[player_name].personal_reward_structure.personal_reward_identity()
        game_value[player_name] = sc.game.ps.unit(Combined(personal=personal, joint=joint))

    game_value_ = frozendict(game_value)
    actions = frozendict()
    return ValueAndActions(game_value=game_value_, mixed_actions=actions)


def solve_final_personal_both(
    sc: SolvingContext[X, U, Y, RP, RJ, SR], gn: GameNode[X, U, Y, RP, RJ, SR]
) -> ValueAndActions[U, RP, RJ]:
    """
    Solves end game node which is final for both players (but not jointly final)
    :param sc:
    :param gn:
    :return:
    """
    game_value: Dict[PlayerName, UncertainCombined] = {}
    for player_name, personal in gn.is_final.items():
        game_value[player_name] = sc.game.ps.unit(Combined(personal=personal, joint=None))
    game_value_ = frozendict(game_value)
    actions = frozendict()
    return ValueAndActions(game_value=game_value_, mixed_actions=actions)


def solve_final_mixed(sc: SolvingContext[X, U, Y, RP, RJ, SR], gn: GameNode[X, U, Y, RP, RJ, SR]
) -> ValueAndActions[U, RP, RJ]:

    game_value: Dict[PlayerName, UncertainCombined] = {}
    game_value.update(
        get_game_values_final(gn=gn, sc=sc)
    )
    game_value_ = frozendict(game_value)
    actions = frozendict()
    return ValueAndActions(game_value=game_value_, mixed_actions=actions)