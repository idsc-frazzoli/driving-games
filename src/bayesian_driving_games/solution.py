import itertools
from collections import defaultdict
from fractions import Fraction
from typing import (
    AbstractSet,
    Dict,
    Mapping,
    Mapping as M,
    Tuple,
    Set,
)

from frozendict import frozendict
from networkx import simple_cycles
from toolz import valmap

from bayesian_driving_games.structures import PlayerType, BayesianGame
from bayesian_driving_games.sequential_rationality import solve_sequential_rationality
from bayesian_driving_games.structures_solution import (
    BayesianGameNode,
    BayesianSolvingContext,
    BayesianGamePreprocessed,
    BayesianGameGraph,
)
from bayesian_driving_games.create_joint_game_tree import create_bayesian_game_graph
from games.solution import fr, get_outcome_preferences_for_players
from possibilities import Poss
from possibilities.sets import SetPoss
from zuper_commons.types import ZValueError
from games import logger
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
    Y,
)
from games.simulate import simulate1, Simulation
from games.structures_solution import (
    GameNode,
    GameSolution,
    Solutions,
    SolutionsPlayer,
    SolvedGameNode,
    SolverParams,
    SolvingContext,
    UsedResources,
    ValueAndActions,
)

__all__ = ["solve_bayesian_game", "get_outcome_preferences_for_players"]


def solve_bayesian_game(gp: BayesianGamePreprocessed) -> Solutions[X, U, Y, RP, RJ, SR]:
    """
    This is the main solving function. However the actual solving algorithm is in solve_game_bayesian2 that is called here.
    This function simulates the results and returns all the solutions.

    :param gp:
    :return: Solution object with strategy, value of the game, etc.
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

    logger.info("creating bayesian game tree")

    # Use game factorization only if the option is set
    if gp.game_factorization and gp.solver_params.use_factorization:
        gf = gp.game_factorization
    else:
        gf = None

    gg = create_bayesian_game_graph(gp.game, gp.solver_params.dt, {initial}, gf=gf)

    game_tree = gg.state2node[initial]
    solutions_players: Dict[PlayerName, SolutionsPlayer[X, U, Y, RP, RJ, SR]] = {}
    initial_state = game_tree.states

    # for player_name, pp in gp.players_pre.items():
    #     # use other solutions
    #     controllers_others = {}
    #     for p2 in gp.players_pre:
    #         for t in gp.game.players[p2].types_of_myself:
    #             if p2 == player_name:
    #                 continue
    #             x_p2 = initial_state[p2]
    #             policy = gp.players_pre[p2].gs.policies[p2 + "," + t]
    #             controllers_others[p2 + "," + t] = AgentFromPolicy(gp.game.ps, policy)

    logger.info("solving bayesian game tree")
    game_solution = solve_game_bayesian2(
        game=gp.game,
        gg=gg,
        solver_params=gp.solver_params,
        jss=initials,
    )
    controllers0 = {}
    for player_name, pp in gp.players_pre.items():
        for typ in gp.game.players[player_name].types_of_myself:
            policy = game_solution.policies[player_name + "," + typ]
            controllers0[player_name, typ] = AgentFromPolicy(gp.game.ps, policy)

    logger.info(
        f"Value of joint solution",
        game_value=game_solution.states_to_solution[initial_state].va.game_value,
        # policy=solution_ghost.policies,
    )
    res = {}
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


def assign_beliefs(sc: BayesianSolvingContext, solution: GameSolution, js: JointState):
    """
    This function takes the strategy from the solution object and updates all the beliefs in the game tree according
    to the formula described in my (Michael's) thesis.
    What is to do yet: The off the path beliefs are at the moment not precise enough.
    At the moment, they do not change to the previous iteration, but actually they can be anything in [0,1].
    This is a recursive function working the tree downwards.

    :param sc: game parameters etc.
    :param solution: Mainly used for the solution strategy.
    :param js: current joint state in order to find the corresponding game node.
    :return: No return, it updates the beliefs in the game tree
    """
    bel: Mapping[PlayerType, Fraction] = {}
    sgn = solution.states_to_solution[js]

    _ = next(iter(sc.game.players))
    type_combinations = list(
        itertools.product(sc.game.players[_].types_of_myself, sc.game.players[_].types_of_others)
    )
    actions_proposed = {}
    for _ in sgn.solved.keys():
        for types in type_combinations:
            actions_proposed[_, types] = Fraction(0, 1)

    for k1, v1 in sgn.va.mixed_actions.items():
        for k2, v2 in sgn.va.mixed_actions.items():
            if (k1[0] != k2[0]) and (k1[0] < k2[0]):  # TODO: Not safe!
                a = frozendict(zip([k1[0], k2[0]], [list(v1.support())[0], list(v2.support())[0]]))
                # dist: Poss[JointPureActions] = game.ps.build_multiple(a=a, f=f)
                types = (k1[2:], k2[2:])
                actions_proposed[a, types] = Fraction(
                    1, 1
                )  # TODO: Mixed strategies need a probability of actions function
    # todo typing not parsing
    gn1: BayesianGameNode = sc.gg.state2node[sgn.states]
    for k, v in sgn.solved.items():
        for p in set(k):
            prior = gn1.game_node_belief[p].support()
            bel = {}
            js2 = list(v.support())[0][p]

            for t in sc.game.players[p].types_of_others:
                for k2, v2 in actions_proposed.items():
                    if (t in k2[1]) and (k == k2[0]):
                        bel[t] = prior[t] * v2

            if sum(bel.values()) != 0:
                factor = Fraction(1, 1) / sum(bel.values())
                for i in bel.keys():
                    bel[i] = bel[i] * factor
                belief = SetPoss(bel)
                sc.gg.state2node[js2].game_node_belief[p] = belief
            else:
                sc.gg.state2node[js2].game_node_belief[p] = gn1.game_node_belief[p]

        assign_beliefs(sc, solution, js2)
    return


def solve_game_bayesian2(
    *,
    game: BayesianGame,
    solver_params: SolverParams,
    gg: BayesianGameGraph,
    jss: AbstractSet[JointState],
) -> GameSolution[X, U, Y, RP, RJ, SR]:
    """
    This is the main solving algorithm as described in my (Michael's) thesis. It consists of a solving step and a belief
    update step. It is an endless loop with the terminal condition that the strategy does not change anymore
    (this is the PBE together with the beliefs).

    :param game:
    :param solver_params:
    :param gg: Game graph with bayesian game nodes.
    :param jss: initiol joint state
    :return: The solution of the Bayesian game.
    """

    outcome_preferences = get_outcome_preferences_for_players(game)
    states_to_solution: Dict[JointState, SolvedGameNode] = {}
    sc = BayesianSolvingContext(
        game=game,
        outcome_preferences=outcome_preferences,
        gg=gg,
        cache=states_to_solution,
        processing=set(),
        solver_params=solver_params,
    )

    x = True
    while x:
        # 1.) solve
        for js0 in jss:
            check_joint_state(js0)
            _solve_bayesian_game(sc, js0)

        policies: Dict[PlayerName, Dict[X, Dict[Poss[JointState], Poss[U]]]]
        ps = game.ps
        policies = defaultdict(lambda: defaultdict(dict))
        for state, s0 in states_to_solution.items():
            for player_name, player_state in state.items():
                for t1 in game.players[player_name].types_of_myself:
                    key = player_name + "," + t1
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
        # 2.) compare strategies
        try:
            solution
        except:
            solution = None

        if solution == solution_new:
            x = False
            return solution_new
        else:
            solution = solution_new

        # 3.) assign beliefs
        for js0 in jss:
            check_joint_state(js0)
            assign_beliefs(sc, solution, js0)

    return None


def add_bayesian_incremental_cost_single(
    game: Game[X, U, Y, RP, RJ, SR],
    *,
    player_name: PlayerName,
    tc: Tuple[PlayerType],
    cur: Combined[RP, RJ],
    incremental_for_player: M[Tuple[PlayerName, Tuple[PlayerType]], Poss[RP]],
) -> Combined[RP, RJ]:
    """
    For every type combination and every player, it updates the cost with the incremental cost.

    :param game:
    :param player_name:
    :param tc: a type cobination
    :param cur:
    :param incremental_for_player: for each player and each type combination an incremental cost
    :return: Returns a updated cost
    """
    try:
        inc = incremental_for_player[player_name, tc]
    except:
        inc = incremental_for_player[player_name, tc[::-1]]
    reduce = game.players[player_name].personal_reward_structure.personal_reward_reduce
    personal = reduce(inc, cur.personal)

    joint = cur.joint
    return Combined(personal=personal, joint=joint)


def _solve_bayesian_game(
    sc: SolvingContext,
    js: JointState,
) -> SolvedGameNode[X, U, Y, RP, RJ, SR]:
    """
    This function is the solving step of the algorithm. It solves the tree backwards for the best strategies and the
    expected game values. Takes the beliefs into account.

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
    type_combinations = list(
        itertools.product(sc.game.players[_].types_of_myself, sc.game.players[_].types_of_other)
    )

    for pure_actions in gn.outcomes:
        # Incremental costs incurred if choosing this action

        inc: Dict[PlayerName, RP]
        try:
            inc = {
                (p, tc): gn.incremental[p, tc][u] for p, u in pure_actions.items() for tc in type_combinations
            }
        except:
            inc = {
                (p, tc[::-1]): gn.incremental[p, tc[::-1]][u]
                for p, u in pure_actions.items()
                for tc in type_combinations
            }
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
                    if (not player_name in gn2.va.game_value) and (
                        not player_name in str(list(gn2.va.game_value.keys()))
                    ):
                        raise ZValueError(player_name=player_name, gn2=gn2, stn=stn)
                    try:
                        return gn2.va.game_value[(player_name, tc)]
                    except:
                        return gn2.va.game_value[(player_name, tc[::-1])]

                stn = solved_to_node[pure_actions]
                # logger.info(stn=stn)
                player_dist: UncertainCombined = ps.join(ps.build(stn, v))

                def f(_: Combined) -> Combined:
                    return add_bayesian_incremental_cost_single(
                        game=sc.game,
                        player_name=player_name,
                        tc=tc,
                        incremental_for_player=inc,
                        cur=_,
                    )

                # logger.info(player_dist=player_dist)
                players_dist[(player_name, tc)] = ps.build(player_dist, f)

        # logger.info(players_dist=players_dist)
        solved[pure_actions] = frozendict(players_dist)

    va: ValueAndActions[U, RP, RJ]
    if gn.joint_final_rewards:  # final costs:
        # fixme: when n > 2, it might be that only part of the crew ends
        va = solve_final_joint_bayesian(sc, gn)
    elif set(gn.states) == set(gn.is_final):
        # All the actives finish independently
        va = solve_final_personal_both_bayesian(sc, gn)
    else:
        va = solve_sequential_rationality(sc, gn, solved)

    # ur: UsedResources[X, U, Y, RP, RJ, SR]
    # usage_current = ps.unit(gn.resources)
    # logger.info(va=va)
    # TODO: Used resources in Bayesian framework...
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
            js=js,
            states=gn.states,
            value=va.game_value,
            processing=len(sc.processing),
            solved=len(sc.cache),
        )
        # logger.info(f"nsolved: {n}")  # , game_value=va.game_value)
    return ret


def solve_final_joint_bayesian(
    sc: SolvingContext[X, U, Y, RP, RJ, SR], gn: GameNode[X, U, Y, RP, RJ, SR]
) -> ValueAndActions[U, RP, RJ]:
    """
    Solves a node which is a joint final node. Game value for each player in each type combination.
    """
    game_value: Dict[PlayerName, UncertainCombined] = {}

    for tc, joint in gn.joint_final_rewards.items():
        for player_name, j in joint.items():
            personal = sc.game.players[player_name].personal_reward_structure.personal_reward_identity()
            game_value[player_name, tc] = sc.game.ps.unit(Combined(personal=personal, joint=j))

    game_value_ = frozendict(game_value)
    actions = frozendict()
    return ValueAndActions(game_value=game_value_, mixed_actions=actions)


def solve_final_personal_both_bayesian(
    sc: SolvingContext[X, U, Y, RP, RJ, SR], gn: GameNode[X, U, Y, RP, RJ, SR]
) -> ValueAndActions[U, RP, RJ]:
    """
    Solves end game node which is final for both players (but not jointly final).
    Game value for each type in each type combination.

    :param sc:
    :param gn:
    :return:
    """
    game_value: Dict[Tuple[PlayerName, Tuple[PlayerType]], UncertainCombined] = {}
    # todo check if this can be done better, I suspect it should go as follows:
    # game_value: M[PlayerName, M[M[PlayerName, PlayerType], UncertainCombined]]
    # for every player I have an outcome for every typecombination of the others
    for player_name, p in gn.is_final.items():
        for tc, personal in p.items():
            game_value[player_name, tc] = sc.game.ps.unit(Combined(personal=personal, joint=None))
    game_value_ = frozendict(game_value)
    actions = frozendict()
    return ValueAndActions(game_value=game_value_, mixed_actions=actions)
