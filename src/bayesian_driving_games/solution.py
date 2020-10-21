import itertools
from collections import defaultdict
from fractions import Fraction
from typing import (
    AbstractSet,
    Dict,
    Mapping,
    Mapping as M,
    Tuple,
    List, Set,
)

from frozendict import frozendict
from networkx import simple_cycles
from toolz import valmap

from bayesian_driving_games import PlayerType, BayesianGame, logger
from bayesian_driving_games.sequential_rationality import solve_sequential_rationality
from bayesian_driving_games.structures_solution import (
    BayesianSolvingContext, BayesianGameNode,
    InformationSet)
from bayesian_driving_games.create_joint_game_tree import create_bayesian_game_graph
from games.solution import add_incremental_cost_single, solve_final_joint, solve_final_personal_both, fr, \
    get_outcome_preferences_for_players
from possibilities import Poss
from possibilities.sets import SetPoss
from zuper_commons.types import ZValueError
from games.agent_from_policy import AgentFromPolicy
from games.game_def import (
    check_joint_state,
    Combined,
    Game,
    JointPureActions,
    JointState,
    PlayerName,
    RJ,
    RP,
    SR,
    U,
    UncertainCombined,
    X,
    Y
)
from games.simulate import simulate1, Simulation
from games.structures_solution import (
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

__all__ = ["solve_bayesian_game"]


def solve_bayesian_game(gp: GamePreprocessed[X, U, Y, RP, RJ, SR]) -> Solutions[X, U, Y, RP, RJ, SR]:
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

    gg = create_bayesian_game_graph(gp.game, gp.solver_params.dt, {initial}, gf=gf)

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

        # if player_name.startswith("N"):  # fixme why?
        #     logger.info(f"looking for solution for {player_name} follower")
        #
        # ghost_game_graph = get_ghost_tree(gp.game, player_name, gg, controllers_others)
        # if player_name.startswith("N"):  # fixme why?
        #     logger.info("The game graph has dimension", nnodes=len(ghost_game_graph.state2node))
        #     # logger.info(gg_nodes=set(ghost_game_graph.state2node))
        #     # logger.info(gg=ghost_game_graph)
        #
        # solution_ghost = solve_game2(
        #     game=gp.game, gg=ghost_game_graph, solver_params=gp.solver_params, jss={initial_state},
        # )
        # msg = f"Stackelberg solution when {player_name} is a follower"
        # game_values = solution_ghost.states_to_solution[initial_state].va.game_value
        # logger.info(msg, game_values=game_values)
        #
        # controllers = dict(controllers_others)
        # controllers[player_name] = AgentFromPolicy(gp.game.ps, solution_ghost.policies[player_name])
        # sim_ = simulate1(gp.game, policies=controllers, initial_states=initial_state, dt=dt, seed=0, )
        # sims[f"{player_name}-follows"] = sim_

    logger.info("solving game tree")
    #initial_strategy, info_sets = proposed_strategy(game=gp.game, gg=gg)
    game_solution = solve_game_bayesian2(
        game=gp.game,
        gg=gg,
        solver_params=gp.solver_params,
        jss=initials,
    )
    controllers0 = {}
    for player_name, pp in gp.players_pre.items():
        for typ in gp.game.players[player_name].types_of_myself:
            policy = game_solution.policies[player_name+','+typ]
            controllers0[player_name, typ] = AgentFromPolicy(gp.game.ps, policy)

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
        game_solution=game_solution, game_tree=game_tree, solutions_players=solutions_players, sims=sims,
    )
    # logger.info(game_tree=game_tree)



### Pro Memoria ###

def proposed_strategy(
    *, game: Game[X, U, Y, RP, RJ, SR], gg: GameGraph[X, U, Y, RP, RJ, SR],
) -> Tuple[Mapping[BayesianGameNode, JointPureActions], Mapping[PlayerName, List[Tuple[JointState]]]]:

    # Step 1: find information sets: For each physical state, the two types together
    players: list = []
    for player_name in game.players:
        players.append(player_name)

    info_sets: Mapping[PlayerName, InformationSet] = {}
    for i in players:
        info_sets[i] = []

    for player_name in game.players:
        active_player = player_name
        inactive_players = [i for i in players if i != player_name]
        if len(inactive_players) != 0:
            for js in gg.state2node:
                if not js[player_name].player_type:
                    msg = "No types in states!"
                    raise ZValueError(msg)

                exists = False
                for js2 in gg.state2node:
                    if js[active_player] == js2[active_player]:  # solve problem here!
                        for i in inactive_players:
                            if js[i].compare_physical_states(js2[i]):
                                exists = True
                                if js[i].player_type < js2[i].player_type:
                                    info_sets[active_player].append((js, js2))
                if exists == False:
                    info_sets[active_player].append((js,))

    for player_name in game.players:
        info_sets[player_name] = set(info_sets[player_name])

    # step 2: Propose a strategy
    proposed_strategy: Mapping[BayesianGameNode, JointPureActions] = {}

    for player_name in players:
        for iset in info_sets[player_name]:
            gn = gg.state2node[iset[0]]
            if gn.moves.values():
                *_, move, _ = gn.moves[player_name]
                for js in iset:
                    # gn = gg.state2node[js]
                    try:
                        proposed_strategy[js][player_name] = move
                    except:
                        proposed_strategy[js] = {player_name: move}

    for _, v in proposed_strategy.items():
        proposed_strategy[_] = frozendict(v)

    return proposed_strategy, info_sets
###

### Pro Memoria ###
def get_initial_physical_states(
    game: Game[X, U, Y, RP, RJ, SR], gg: GameGraph[X, U, Y, RP, RJ, SR],
) -> List[JointState]:

    initial = next(iter(gg.initials))
    gn = gg.state2node[initial]

    initial_physicals = []
    for move in gn.outcomes:
        helper1 = gn.outcomes[move].support()
        (helper2,) = helper1
        js_new = helper2[next(iter(game.players))]
        initial_physicals.append(js_new)

    return initial_physicals
###

# TODO
def assign_beliefs(
    sc: BayesianSolvingContext,
    solution: GameSolution,
    js: JointState
):
    bel: Mapping[PlayerType, Fraction] = {}
    sgn = solution.states_to_solution[js]

    _ = next(iter(sc.game.players))
    type_combinations = list(itertools.product(sc.game.players[_].types_of_myself, sc.game.players[_].types_of_other))
    actions_proposed = {}
    for _ in sgn.solved.keys():
        for types in type_combinations:
            actions_proposed[_,types] = Fraction(0,1)

    for k1,v1 in sgn.va.mixed_actions.items():
        for k2,v2 in sgn.va.mixed_actions.items():
            if (k1[0]!=k2[0]) and (k1[0]<k2[0]): #TODO: Not safe!
                a = frozendict(zip([k1[0], k2[0]], [list(v1.support())[0], list(v2.support())[0]]))
                # dist: Poss[JointPureActions] = game.ps.build_multiple(a=a, f=f)
                types = (k1[2:],k2[2:])
                actions_proposed[a,types] = Fraction(1,1) #TODO: Mixed strategies need a probability of actions function

    gn1: BayesianGameNode = sc.gg.state2node[sgn.states]
    for k,v in sgn.solved.items():
        for p in sc.game.players:
            prior = gn1.game_node_belief[p].support()
            bel = {}
            js2 = list(v.support())[0][p]

            for t in sc.game.players[p].types_of_other:
                for k2, v2 in actions_proposed.items():
                    if (t in k2[1]) and (k == k2[0]):
                        bel[t] = prior[t]*v2

            try:
                factor = Fraction(1, 1) / sum(bel.values())
            except:
                factor = Fraction(0, 1)
            for i in bel.keys():
                bel[i] = bel[i] * factor

            belief = SetPoss(bel)
            sc.gg.state2node[js2].game_node_belief[p] = belief

        assign_beliefs(sc, solution, js2)

    return


def solve_game_bayesian2(
    *,
    game: BayesianGame[X, U, Y, RP, RJ, SR],
    solver_params: SolverParams,
    gg: GameGraph[X, U, Y, RP, RJ, SR],
    jss: AbstractSet[JointState],
) -> GameSolution[X, U, Y, RP, RJ, SR]:

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
    #
    # initial_physicals = get_initial_physical_states(game, gg)
    # for js in initial_physicals:
    #     for player_name in game.players:
    #         assign_beliefs(game, gg, info_sets, js, sc, player_name)

    x = True
    while x:
        #1.) solve
        for js0 in jss:
            check_joint_state(js0)
            _solve_bayesian_game(sc, js0)

        policies: Dict[PlayerName, Dict[X, Dict[Poss[JointState], Poss[U]]]]
        ps = game.ps
        policies = defaultdict(lambda: defaultdict(dict))
        for state, s0 in states_to_solution.items():
            for player_name, player_state in state.items():
                for t1 in game.players[player_name].types_of_myself:
                    key = player_name+","+t1
                    if key in s0.va.mixed_actions:
                        policy_for_this_state = policies[key][player_state]
                        other_states = frozendict({k: v for k, v in state.items() if k != player_name})
                        iset = ps.unit(other_states)
                        policy_for_this_state[iset] = s0.va.mixed_actions[key]

        policies2 = frozendict({k: fr(v) for k, v in policies.items()})

        solution_new = GameSolution(
            initials=frozenset(jss),
            policies=policies2,
            states_to_solution=frozendict(states_to_solution),
        )
        #2.) compare strategies
        try:
            solution
        except:
            solution = None

        if solution == solution_new:
            x = False
            return solution_new
        else:
            solution = solution_new

        #3.) assign beliefs
        for js0 in jss:
            check_joint_state(js0)
            assign_beliefs(sc, solution, js0)





    return None


# def solve_game2(
#     *,
#     game: Game[X, U, Y, RP, RJ, SR],
#     solver_params: SolverParams,
#     gg: GameGraph[X, U, Y, RP, RJ, SR],
#     jss: AbstractSet[JointState],
# ) -> GameSolution[X, U, Y, RP, RJ, SR]:
#     """
#     Solve game
#     :param game:
#     :param solver_params:
#     :param gg:
#     :param jss:
#     :return:
#     """
#     outcome_preferences = get_outcome_preferences_for_players(game)
#     states_to_solution: Dict[JointState, SolvedGameNode] = {}
#     sc = SolvingContext(
#         game=game,
#         outcome_preferences=outcome_preferences,
#         gg=gg,
#         cache=states_to_solution,
#         processing=set(),
#         solver_params=solver_params,
#     )
#     for js0 in jss:
#         check_joint_state(js0)
#         _solve_game(sc, js0)
#
#     policies: Dict[PlayerName, Dict[X, Dict[Poss[JointState], Poss[U]]]]
#     ps = game.ps
#     policies = defaultdict(lambda: defaultdict(dict))
#     for state, s0 in states_to_solution.items():
#         for player_name, player_state in state.items():
#
#             if player_name in s0.va.mixed_actions:
#                 policy_for_this_state = policies[player_name][player_state]
#                 other_states = frozendict({k: v for k, v in state.items() if k != player_name})
#                 iset = ps.unit(other_states)
#                 policy_for_this_state[iset] = s0.va.mixed_actions[player_name]
#
#     policies2 = frozendict({k: fr(v) for k, v in policies.items()})
#
#     return GameSolution(
#         initials=frozenset(jss), policies=policies2, states_to_solution=frozendict(states_to_solution),
#     )



def _solve_bayesian_game(sc: SolvingContext, js: JointState,) -> SolvedGameNode[X, U, Y, RP, RJ, SR]:
    """
    # Actual recursive function that solves the game nodes
    :param sc:
    :param js:
    :return:
    """
    check_joint_state(js)
    if not js:
        raise ZValueError(js=js)
    if js in sc.cache:
        return sc.cache[js]
    if js in sc.processing:
        msg = "Loop found"
        raise ZValueError(msg, states=js)
    gn: BayesianGameNode[X, U, Y, RP, RJ, SR] = sc.gg.state2node[js]
    sc.processing.add(js)

    ps = sc.game.ps
    # what happens for each pure action?
    pure_actions: JointPureActions
    solved: Dict[JointPureActions, M[Tuple[PlayerName, PlayerType], UncertainCombined]] = {}
    solved_to_node: Dict[JointPureActions, Poss[M[PlayerName, JointState]]]
    solved_to_node = {}

    _ = next(iter(sc.game.players))
    type_combinations = list(itertools.product(sc.game.players[_].types_of_myself, sc.game.players[_].types_of_other))

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
            return frozendict(valmap(lambda _: _solve_bayesian_game(sc, _).states, a))

        solved_to_node[pure_actions] = ps.build(next_nodes, u)

        players_dist: Dict[Set[PlayerName, PlayerType], UncertainCombined] = {}
        for player_name in pure_actions:
            for tc in type_combinations:

                def v(m: M[PlayerName, JointState]) -> UncertainCombined:
                    gn2: SolvedGameNode[X, U, U, RP, RJ, SR] = sc.cache[m[player_name]]
                    if (not player_name in gn2.va.game_value) and (not player_name in str(list(gn2.va.game_value.keys()))):
                        raise ZValueError(player_name=player_name, gn2=gn2, stn=stn)
                    return gn2.va.game_value[(player_name,tc)]

                stn = solved_to_node[pure_actions]
                # logger.info(stn=stn)
                player_dist: UncertainCombined = ps.join(ps.build(stn, v))

                def f(_: Combined) -> Combined:
                    return add_incremental_cost_single(
                        game=sc.game, player_name=player_name, incremental_for_player=inc, cur=_,
                    )

                # logger.info(player_dist=player_dist)
                players_dist[(player_name,tc)] = ps.build(player_dist, f)

        # logger.info(players_dist=players_dist)
        solved[pure_actions] = frozendict(players_dist)

    va: ValueAndActions[U, RP, RJ]
    if gn.joint_final_rewards:  # final costs:
        # fixme: when n > 2, it might be that only part of the crew ends
        va = solve_final_joint_bayesian(sc, gn)
    elif set(gn.states) == set(gn.is_final):
        # All the actives finish independently
        va = solve_final_personal_both(sc, gn)
    else:
        va = solve_sequential_rationality(sc, gn, solved)

    ur: UsedResources[X, U, Y, RP, RJ, SR]
    usage_current = ps.unit(gn.resources)
    # logger.info(va=va)
    #TODO: Used resources in Bayesian framework...
    # if va.mixed_actions:  # not a final state
    #     next_states: Poss[M[PlayerName, SolvedGameNode[X, U, U, RP, RJ, SR]]]
    #     next_states = ps.join(ps.build_multiple(va.mixed_actions, solved_to_node.__getitem__))
    #
    #     usages: Dict[D, Poss[M[PlayerName, FrozenSet[SR]]]]
    #     usages = {D(0): usage_current}
    #     Π = 1
    #
    #     for i in map(D, range(10)):  # XXX: use the range that's needed
    #         default = ps.unit(frozendict())
    #
    #         def get_data(x: M[PlayerName, JointState]) -> Poss[Mapping[PlayerName, FSet[SR]]]:
    #             used_by_players: Dict[PlayerName, Poss[FSet[SR]]] = {}
    #             for pname in va.mixed_actions:
    #
    #                 def get_its(y: Mapping[PlayerName, FSet[SR]]) -> FSet[SR]:
    #                     return y.get(pname, frozenset())
    #
    #                 st = x[player_name]
    #                 gn_ = sc.cache[st]
    #                 ui = gn_.ur.used.get(i, default)
    #                 used_at_i_by_player: Poss[FSet[SR]] = ps.build(ui, get_its)
    #                 used_by_players[pname] = used_at_i_by_player
    #
    #             def remove_empty(_: Mapping[PlayerName, FSet[SR]]) -> Mapping[PlayerName, FSet[SR]]:
    #                 notempty = {}
    #                 for k, sr_used in _.items():
    #                     if sr_used:
    #                         notempty[k] = sr_used
    #                 return frozendict(notempty)
    #
    #             res: Poss[Mapping[PlayerName, FSet[SR]]]
    #             res = ps.build_multiple(used_by_players, remove_empty)
    #             return res
    #
    #         at_d = ps.build(next_states, get_data)
    #         f = ps.join(at_d)
    #         if f.support() != {frozendict()}:
    #             usages[i + 1] = f
    #
    #     # logger.info(next_resources=next_resources,
    #     #             usages=usages)
    #
    #     ur = UsedResources(frozendict(usages))
    # else:
    #
    #     usages_ = frozendict({D(0): usage_current})
    #     ur = UsedResources(usages_)

    try:
        ret = SolvedGameNode(states=js, solved=frozendict(solved_to_node), va=va, ur=None)
    except Exception as e:
        raise ZValueError(game_node=gn) from e
    sc.cache[js] = ret
    sc.processing.remove(js)

    n = len(sc.cache)
    if n % 30 == 0:
        logger.info(
            js=js, states=gn.states, value=va.game_value, processing=len(sc.processing), solved=len(sc.cache),
        )
        # logger.info(f"nsolved: {n}")  # , game_value=va.game_value)
    return ret


# def _solve_game(
#     sc: SolvingContext[X, U, Y, RP, RJ, SR], js: JointState,
# ) -> SolvedGameNode[X, U, Y, RP, RJ, SR]:
#     """
#     # Actual recursive function that solves the game nodes
#     :param sc:
#     :param js:
#     :return:
#     """
#     check_joint_state(js)
#     if not js:
#         raise ZValueError(js=js)
#     if js in sc.cache:
#         return sc.cache[js]
#     if js in sc.processing:
#         msg = "Loop found"
#         raise ZValueError(msg, states=js)
#     gn: GameNode[X, U, Y, RP, RJ, SR] = sc.gg.state2node[js]
#     sc.processing.add(js)
#
#     ps = sc.game.ps
#     # what happens for each pure action?
#     pure_actions: JointPureActions
#     solved: Dict[JointPureActions, M[PlayerName, UncertainCombined]] = {}
#     solved_to_node: Dict[JointPureActions, Poss[M[PlayerName, JointState]]]
#     solved_to_node = {}
#
#     for pure_actions in gn.outcomes:
#         # Incremental costs incurred if choosing this action
#         inc: Dict[PlayerName, RP]
#         inc = {p: gn.incremental[p][u] for p, u in pure_actions.items()}
#         # if we choose these actions, then these are the game nodes
#         # we could go in. Note that each player can go in a different joint state.
#         next_nodes: Poss[Mapping[PlayerName, JointState]] = gn.outcomes[pure_actions]
#
#         # These are the solved nodes; for each, we find the solutions (recursive step here)
#         # def u(a: M[PlayerName, JointState]) -> M[PlayerName, SolvedGameNode[X, U, U, RP, RJ, SR]]:
#         def u(a: M[PlayerName, JointState]) -> M[PlayerName, JointState]:
#             return frozendict(valmap(lambda _: _solve_game(sc, _).states, a))
#
#         solved_to_node[pure_actions] = ps.build(next_nodes, u)
#
#         players_dist: Dict[PlayerName, UncertainCombined] = {}
#         for player_name in pure_actions:
#
#             def v(m: M[PlayerName, JointState]) -> UncertainCombined:
#                 gn2: SolvedGameNode[X, U, U, RP, RJ, SR] = sc.cache[m[player_name]]
#                 if not player_name in gn2.va.game_value:
#                     raise ZValueError(player_name=player_name, gn2=gn2, stn=stn)
#                 return gn2.va.game_value[player_name]
#
#             stn = solved_to_node[pure_actions]
#             # logger.info(stn=stn)
#             player_dist: UncertainCombined = ps.join(ps.build(stn, v))
#
#             def f(_: Combined) -> Combined:
#                 return add_incremental_cost_single(
#                     game=sc.game, player_name=player_name, incremental_for_player=inc, cur=_,
#                 )
#
#             # logger.info(player_dist=player_dist)
#             players_dist[player_name] = ps.build(player_dist, f)
#
#         # logger.info(players_dist=players_dist)
#         solved[pure_actions] = frozendict(players_dist)
#
#     va: ValueAndActions[U, RP, RJ]
#     if gn.joint_final_rewards:  # final costs:
#         # fixme: when n > 2, it might be that only part of the crew ends
#         va = solve_final_joint(sc, gn)
#     elif set(gn.states) == set(gn.is_final):
#         # All the actives finish independently
#         va = solve_final_personal_both(sc, gn)
#     else:
#         va = solve_equilibria(sc, gn, solved)
#
#     ur: UsedResources[X, U, Y, RP, RJ, SR]
#     usage_current = ps.unit(gn.resources)
#     # logger.info(va=va)
#     if va.mixed_actions:  # not a final state
#         next_states: Poss[M[PlayerName, SolvedGameNode[X, U, U, RP, RJ, SR]]]
#         next_states = ps.join(ps.build_multiple(va.mixed_actions, solved_to_node.__getitem__))
#
#         usages: Dict[D, Poss[M[PlayerName, FrozenSet[SR]]]]
#         usages = {D(0): usage_current}
#         Π = 1
#
#         for i in map(D, range(10)):  # XXX: use the range that's needed
#             default = ps.unit(frozendict())
#
#             def get_data(x: M[PlayerName, JointState]) -> Poss[Mapping[PlayerName, FSet[SR]]]:
#                 used_by_players: Dict[PlayerName, Poss[FSet[SR]]] = {}
#                 for pname in va.mixed_actions:
#
#                     def get_its(y: Mapping[PlayerName, FSet[SR]]) -> FSet[SR]:
#                         return y.get(pname, frozenset())
#
#
#                 def remove_empty(_: Mapping[PlayerName, FSet[SR]]) -> Mapping[PlayerName, FSet[SR]]:
#                     notempty = {}
#                     for k, sr_used in _.items():
#                         if sr_used:
#                             notempty[k] = sr_used
#                     return frozendict(notempty)
#
#                 res: Poss[Mapping[PlayerName, FSet[SR]]]
#                 res = ps.build_multiple(used_by_players, remove_empty)
#                 return res
#
#             at_d = ps.build(next_states, get_data)
#             f = ps.join(at_d)
#             if f.support() != {frozendict()}:
#                 usages[i + 1] = f
#
#         # logger.info(next_resources=next_resources,
#         #             usages=usages)
#
#         ur = UsedResources(frozendict(usages))
#     else:
#
#         usages_ = frozendict({D(0): usage_current})
#         ur = UsedResources(usages_)
#
#     try:
#         ret = SolvedGameNode(states=js, solved=frozendict(solved_to_node), va=va, ur=ur)
#     except Exception as e:
#         raise ZValueError(game_node=gn) from e
#     sc.cache[js] = ret
#     sc.processing.remove(js)
#
#     n = len(sc.cache)
#     if n % 30 == 0:
#         logger.info(
#             js=js, states=gn.states, value=va.game_value, processing=len(sc.processing), solved=len(sc.cache),
#         )
#         # logger.info(f"nsolved: {n}")  # , game_value=va.game_value)
#     return ret



def solve_final_joint_bayesian(
    sc: SolvingContext[X, U, Y, RP, RJ, SR], gn: GameNode[X, U, Y, RP, RJ, SR]
) -> ValueAndActions[U, RP, RJ]:
    """
    Solves a node which is a joint final node
    """
    game_value: Dict[PlayerName, UncertainCombined] = {}

    for tc, joint in gn.joint_final_rewards.items():
        for player_name, j in joint.items():
            personal = sc.game.players[player_name].personal_reward_structure.personal_reward_identity()
            game_value[player_name, tc] = sc.game.ps.unit(Combined(personal=personal, joint=j))

    game_value_ = frozendict(game_value)
    actions = frozendict()
    return ValueAndActions(game_value=game_value_, mixed_actions=actions)


