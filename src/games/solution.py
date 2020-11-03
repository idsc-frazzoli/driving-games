from collections import defaultdict
from decimal import Decimal as D
from typing import (
    AbstractSet,
    Callable,
    Dict,
    FrozenSet,
    FrozenSet as FSet,
    Mapping,
    Mapping as M,
)

from frozendict import frozendict
from networkx import simple_cycles
from toolz import valmap

from possibilities import Poss
from preferences import Preference
from zuper_commons.types import ZValueError
from . import logger
from .agent_from_policy import AgentFromPolicy
from .create_joint_game_tree import create_game_graph
from .game_def import (
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
    MonadicPreferenceBuilder,
)
from .simulate import simulate1, Simulation
from .solution_ghost import get_ghost_tree
from .solve_equilibria_ import solve_equilibria
from .structures_solution import (
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

__all__ = ["solve1", "get_outcome_preferences_for_players"]


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

    for player_name, pp in gp.players_pre.items():
        # use other solutions
        controllers_others = {}
        for p2 in gp.players_pre:
            if p2 == player_name:
                continue
            x_p2 = initial_state[p2]
            policy = gp.players_pre[p2].gs.policies[p2]
            controllers_others[p2] = AgentFromPolicy(gp.game.ps, policy)

        if player_name.startswith("N"):  # fixme why?
            logger.info(f"looking for solution for {player_name} follower")

        ghost_game_graph = get_ghost_tree(gp.game, player_name, gg, controllers_others)
        if player_name.startswith("N"):  # fixme why?
            logger.info("The game graph has dimension", nnodes=len(ghost_game_graph.state2node))
            # logger.info(gg_nodes=set(ghost_game_graph.state2node))
            # logger.info(gg=ghost_game_graph)

        solution_ghost = solve_game2(
            game=gp.game,
            gg=ghost_game_graph,
            solver_params=gp.solver_params,
            jss={initial_state},
        )
        msg = f"Stackelberg solution when {player_name} is a follower"
        game_values = solution_ghost.states_to_solution[initial_state].va.game_value
        logger.info(msg, game_values=game_values)

        controllers = dict(controllers_others)
        controllers[player_name] = AgentFromPolicy(gp.game.ps, solution_ghost.policies[player_name])
        sim_ = simulate1(
            gp.game,
            policies=controllers,
            initial_states=initial_state,
            dt=dt,
            seed=0,
        )
        sims[f"{player_name}-follows"] = sim_

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


def get_outcome_preferences_for_players(
    game: Game[X, U, Y, RP, RJ, SR],
) -> M[PlayerName, Preference[UncertainCombined]]:
    """

    :param game:
    :return:
    """
    preferences: Dict[PlayerName, Preference[UncertainCombined]] = {}
    for player_name, player in game.players.items():
        pref0: Preference[Combined[RJ, RP]] = player.preferences
        monadic_pref_builder: MonadicPreferenceBuilder
        monadic_pref_builder = player.monadic_preference_builder
        pref2: Preference[UncertainCombined] = monadic_pref_builder(pref0)
        preferences[player_name] = pref2
    return preferences


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
    # Instead of X it should be something like InfoSet or GameNode?!
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


def fr(d):
    return frozendict({k: frozendict(v) for k, v in d.items()})


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
        # if we choose these actions, then these are the game nodes
        # we could go in. Note that each player can go in a different joint state.
        next_nodes: Poss[Mapping[PlayerName, JointState]] = gn.outcomes[pure_actions]

        # These are the solved nodes; for each, we find the solutions (recursive step here)
        # def u(a: M[PlayerName, JointState]) -> M[PlayerName, SolvedGameNode[X, U, U, RP, RJ, SR]]:
        def u(a: M[PlayerName, JointState]) -> M[PlayerName, JointState]:
            return frozendict(valmap(lambda _: _solve_game(sc, _).states, a))

        solved_to_node[pure_actions] = ps.build(next_nodes, u)

        players_dist: Dict[PlayerName, UncertainCombined] = {}
        for player_name in pure_actions:

            def v(m: Mapping[PlayerName, JointState]) -> UncertainCombined:
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
    if gn.joint_final_rewards:  # final costs:
        # fixme: when n > 2, it might be that only part of the crew ends
        va = solve_final_joint(sc, gn)
    elif set(gn.states) == set(gn.is_final):
        # All the actives finish independently
        va = solve_final_personal_both(sc, gn)
    else:
        va = solve_equilibria(sc, gn, solved)

    ur: UsedResources[X, U, Y, RP, RJ, SR]
    usage_current = ps.unit(gn.resources)
    # logger.info(va=va)
    if va.mixed_actions:
        next_states: Poss[M[PlayerName, SolvedGameNode[X, U, U, RP, RJ, SR]]]
        next_states = ps.join(ps.build_multiple(va.mixed_actions, solved_to_node.__getitem__))

        usages: Dict[D, Poss[M[PlayerName, FrozenSet[SR]]]]
        usages = {D(0): usage_current}
        Π = 1

        for i in map(D, range(10)):  # XXX: use the range that's needed
            default = ps.unit(frozendict())

            def get_data(x: M[PlayerName, JointState]) -> Poss[Mapping[PlayerName, FSet[SR]]]:
                used_by_players: Dict[PlayerName, Poss[FSet[SR]]] = {}
                for pname in va.mixed_actions:

                    def get_its(y: Mapping[PlayerName, FSet[SR]]) -> FSet[SR]:
                        return y.get(pname, frozenset())

                    st = x[player_name]
                    gn_ = sc.cache[st]
                    ui = gn_.ur.used.get(i, default)
                    used_at_i_by_player: Poss[FSet[SR]] = ps.build(ui, get_its)
                    used_by_players[pname] = used_at_i_by_player

                def remove_empty(_: Mapping[PlayerName, FSet[SR]]) -> Mapping[PlayerName, FSet[SR]]:
                    notempty = {}
                    for k, sr_used in _.items():
                        if sr_used:
                            notempty[k] = sr_used
                    return frozendict(notempty)

                res: Poss[Mapping[PlayerName, FSet[SR]]]
                res = ps.build_multiple(used_by_players, remove_empty)
                return res

            at_d = ps.build(next_states, get_data)
            f = ps.join(at_d)
            if f.support() != {frozendict()}:
                usages[i + 1] = f

        # logger.info(next_resources=next_resources,
        #             usages=usages)

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
        logger.info(
            js=js,
            states=gn.states,
            value=va.game_value,
            processing=len(sc.processing),
            solved=len(sc.cache),
        )
        # logger.info(f"nsolved: {n}")  # , game_value=va.game_value)
    return ret


def add_incremental_cost_single(
    game: Game[X, U, Y, RP, RJ, SR],
    *,
    player_name: PlayerName,
    cur: Combined[RP, RJ],
    incremental_for_player: M[PlayerName, Poss[RP]],
) -> Combined[RP, RJ]:
    inc = incremental_for_player[player_name]
    reduce = game.players[player_name].personal_reward_structure.personal_reward_reduce
    personal = reduce(inc, cur.personal)

    joint = cur.joint
    return Combined(personal=personal, joint=joint)


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
