from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal as D
from typing import AbstractSet, Callable, Dict, FrozenSet, Generic, Mapping

from frozendict import frozendict
from networkx import simple_cycles

from possibilities import Poss
from preferences import PrefConverter, Preference
from zuper_commons.types import check_isinstance, ZValueError
from . import logger
from .agent_from_policy import AgentFromPolicy
from .create_joint_game_tree import create_game_graph
from .game_def import (
    check_joint_state,
    Combined,
    Game,
    JointPureActions,
    JointState,
    Outcome,
    P,
    PlayerName,
    Pr,
    RJ,
    RP,
    SetOfOutcomes,
    SR,
    U,
    X,
    Y,
)
from .simulate import simulate1, Simulation
from .solution_ghost import get_ghost_tree
from .solve_equilibria_ import solve_equilibria
from .structures_solution import (
    check_set_outcomes,
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

__all__ = ["solve1", "get_outcome_set_preferences_for_players"]


def solve1(gp: GamePreprocessed[Pr, X, U, Y, RP, RJ, SR]) -> Solutions[Pr, X, U, Y, RP, RJ, SR]:
    G = gp.game_graph

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

    # cache = {}

    logger.info("creating game tree")
    gg = create_game_graph(gp.game, gp.solver_params.dt, {initial})

    game_tree = gg.state2node[initial]
    solutions_players: Dict[PlayerName, SolutionsPlayer[Pr, X, U, Y, RP, RJ, SR]] = {}
    initial_state = game_tree.states
    # alone_solutions: Dict[PlayerName, Dict[X, GameSolution]] = {}
    # for player_name, pp in gp.players_pre.items():
    #
    #     alone_solutions[player_name] = {}
    #     for x0, personal_tree in pp.alone_tree.items():
    #         solved_x0 = solve_game2(gp, gg, x0)
    #         alone_solutions[player_name][x0] = solved_x0
    #         logger.info(
    #             f"Solution for {player_name} alone",
    #             game_value=solved_x0.gn_solved.va.game_value,
    #             # policy=solved_x0.policies
    #         )

    for player_name, pp in gp.players_pre.items():
        # use other solutions
        # logger.info("looking for ghost solutions")
        controllers_others = {}
        for p2 in gp.players_pre:
            if p2 == player_name:
                continue
            x_p2 = initial_state[p2]
            policy = gp.players_pre[p2].gs.policies[p2]
            # alone_solutions_p2 = alone_solutions[p2]

            # policy = alone_solutions_p2[x_p2].policies[p2]
            controllers_others[p2] = AgentFromPolicy(policy)

        ghost_game_graph = get_ghost_tree(gp.game, player_name, gg, controllers_others)

        # player_start = list(pp.game_graph.initials)[0]
        # player_start = frozendict({player_name: x0})
        # logger.info(
        #     "first node of tree ghost",
        #     tree_ghost=replace(tree_ghost, outcomes=frozendict()),
        #     outcomes=set(tree_ghost.outcomes),
        # )
        solution_ghost = solve_game2(
            game=gp.game, gg=ghost_game_graph, solver_params=gp.solver_params, jss={initial_state}
        )
        logger.info(
            f"Stackelberg solution when {player_name} is a follower",
            game_value=solution_ghost.states_to_solution[initial_state].va.game_value,
            # policy=solution_ghost.policies,
        )
        controllers = dict(controllers_others)
        controllers[player_name] = AgentFromPolicy(solution_ghost.policies[player_name])
        sim_ = simulate1(
            gp.game, policies=controllers, initial_states=initial_state, dt=gp.solver_params.dt, seed=0,
        )
        sims[f"{player_name}-follows"] = sim_

    logger.info("solving game tree")
    game_solution = solve_game2(game=gp.game, solver_params=gp.solver_params, gg=gg, jss=initials)
    controllers0 = {}
    for player_name, pp in gp.players_pre.items():
        policy = game_solution.policies[player_name]
        controllers0[player_name] = AgentFromPolicy(policy)

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


def get_outcome_set_preferences_for_players(
    game: Game[Pr, X, U, Y, RP, RJ, SR],
) -> Mapping[PlayerName, Preference[SetOfOutcomes]]:
    preferences = {}
    for player_name, player in game.players.items():
        # Comparse Combined[RJ, RP]
        pref0: Preference[Combined[RJ, RP]] = player.preferences
        pref1: Preference[Outcome[RJ, RP]]
        pref1 = PrefConverter(AT=Outcome, BT=Combined, convert=CombinedFromOutcome(player_name), p0=pref0)
        # compares Aset(Combined[RJ, RP]
        set_preference_aggregator: Callable[[Preference[P]], Preference[Poss[P, Pr]]]
        set_preference_aggregator = player.set_preference_aggregator
        pref2: Preference[SetOfOutcomes] = set_preference_aggregator(pref1)
        # result: Preference[ASet[Outcome[RP, RJ]]]
        preferences[player_name] = pref2
    return preferences


def solve_game2(
    *,
    game: Game[Pr, X, U, Y, RP, RJ, SR],
    solver_params: SolverParams,
    gg: GameGraph[Pr, X, U, Y, RP, RJ, SR],
    jss: AbstractSet[JointState],
) -> GameSolution[Pr, X, U, Y, RP, RJ, SR]:
    outcome_set_preferences = get_outcome_set_preferences_for_players(game)
    states_to_solution: Dict[JointState, SolvedGameNode] = {}
    sc = SolvingContext(
        game=game,
        outcome_set_preferences=outcome_set_preferences,
        gg=gg,
        cache=states_to_solution,
        processing=set(),
        solver_params=solver_params,
    )
    for js0 in jss:
        check_joint_state(js0)
        _solve_game(sc, js0)

    policies: Dict[PlayerName, Dict[X, Dict[Poss[JointState, Pr], Poss[U, Pr]]]]
    ps = game.ps
    policies = defaultdict(lambda: defaultdict(dict))
    for state, s0 in states_to_solution.items():
        for player_name, player_state in state.items():

            if player_name in s0.va.mixed_actions:
                policy_for_this_state = policies[player_name][player_state]
                other_states = frozendict({k: v for k, v in state.items() if k != player_name})
                iset = ps.lift_one(other_states)  # frozenset({other_states})
                policy_for_this_state[iset] = s0.va.mixed_actions[player_name]

    # policies2 = frozendict(valmap(policies, )
    policies2 = frozendict({k: fr(v) for k, v in policies.items()})

    return GameSolution(
        initials=frozenset(jss), policies=policies2, states_to_solution=frozendict(states_to_solution)
    )


def fr(d):
    return frozendict({k: frozendict(v) for k, v in d.items()})


def _solve_game(
    sc: SolvingContext[Pr, X, U, Y, RP, RJ, SR], js: JointState,
) -> SolvedGameNode[Pr, X, U, Y, RP, RJ, SR]:
    check_joint_state(js)
    if not js:
        raise ZValueError(js=js)
    if js in sc.cache:
        return sc.cache[js]
    if js in sc.processing:
        msg = "Loop found"
        raise ZValueError(msg, states=js)
    gn: GameNode[Pr, X, U, Y, RP, RJ] = sc.gg.state2node[js]
    sc.processing.add(js)

    # logger.debug(gn_states=gn.states, processing=len(sc.processing),
    #              processed=len(sc.cache))
    # for pure_actions in gn.outcomes2:
    #     check_joint_pure_actions(pure_actions)

    ps = sc.game.ps
    # what happens for each action?
    pure_actions: JointPureActions
    solved: Dict[JointPureActions, SetOfOutcomes] = {}
    solved_to_node: Dict[JointPureActions, Poss[SolvedGameNode[Pr, X, U, U, RP, RJ], Pr]] = {}

    for pure_actions in gn.outcomes3:
        # Incremental costs incurred if choosing this action
        inc: Dict[PlayerName, RP]
        inc = {p: gn.incremental[p][u] for p, u in pure_actions.items()}
        # if we choose these actions, then these are the game nodes
        # we could go in
        next_nodes2: Poss[JointState, Pr] = gn.outcomes3[pure_actions]
        # logger.info(gn_outcomes3=gn.outcomes3,pure_actions=pure_actions, next_nodes2=next_nodes2)
        # and these are the solved nodes (recursive step here)
        next_nodes_solutions: Poss[SolvedGameNode[Pr, X, U, U, RP, RJ], Pr]
        next_nodes_solutions = ps.build(next_nodes2, lambda _: _solve_game(sc, _))
        solved_to_node[pure_actions] = next_nodes_solutions
        # For each, we find the solution
        next_outcomes_solutions: Poss[SetOfOutcomes, Pr]
        next_outcomes_solutions = ps.build(next_nodes_solutions, lambda _: _.va.game_value)
        # and now we flatten
        flattened: SetOfOutcomes = ps.flatten(next_outcomes_solutions)
        # check_set_outcomes(flattened)
        # Now we need to add the incremental costs
        f = lambda _: add_incremental_cost(game=sc.game, incremental_for_player=inc, outcome=_)
        # logger.info(flattened=flattened)
        added: SetOfOutcomes = ps.build(flattened, f)

        # check_set_outcomes(added)
        solved[pure_actions] = added

    va: ValueAndActions[Pr, U, RP, RJ]
    if gn.joint_final_rewards:  # final costs:
        va = solve_final_joint(sc, gn)
    elif set(gn.states) == set(gn.is_final):
        # They both finished
        va = solve_final_personal_both(sc, gn)
    else:
        va = solve_equilibria(sc, gn, solved)

    # logger.info(va=va)
    if va.mixed_actions:  # not a final state
        next_states: Poss[SolvedGameNode[Pr, X, U, U, RP, RJ], Pr]
        next_states = ps.flatten(ps.build_multiple(va.mixed_actions, solved_to_node.__getitem__))

        next_resources: Poss[UsedResources, Pr]
        next_resources = ps.build(next_states, lambda _: _.ur)
        # usages: Dict[D, Poss[Mapping[PlayerName, FrozenSet[SR]], Pr]]
        # logger.info(next_resources=next_resources)
        usage_current = ps.lift_one(gn.resources)
        usages: Dict[D, Poss[Mapping[PlayerName, FrozenSet[SR]], Pr]]
        usages = {D(0): usage_current}

        for i in range(10):
            default = ps.lift_one(frozendict())
            at_d = ps.build(next_resources, lambda r: r.used.get(i, default))
            f = ps.flatten(at_d)
            if f.support() != {frozendict()}:
                usages[D(i + 1)] = f

        # logger.info(next_resources=next_resources,
        #             usages=usages)
        # ur: UsedResources[Pr, X, U, Y, RP, RJ, SR]
        ur = UsedResources(frozendict(usages))
    else:
        ur = UsedResources(frozendict())

    ret = SolvedGameNode(gn=gn, solved=frozendict(solved_to_node), va=va, ur=ur)
    sc.cache[js] = ret
    sc.processing.remove(js)

    n = len(sc.cache)
    if n % 30 == 0:
        logger.info(
            js=js, states=gn.states, value=va.game_value, processing=len(sc.processing), solved=len(sc.cache)
        )
        # logger.info(f"nsolved: {n}")  # , game_value=va.game_value)
    return ret


def add_incremental_cost(
    game: Game[Pr, X, U, Y, RP, RJ, SR],
    *,
    outcome: Outcome[RP, RJ],
    incremental_for_player: Mapping[PlayerName, RP],
) -> Outcome[RP, RJ]:
    check_isinstance(outcome, Outcome)
    private = {}
    # logger.info(outcome=outcome, action=action, incremental=incremental)
    u: U
    for player_name, inc in incremental_for_player.items():
        reduce = game.players[player_name].personal_reward_structure.personal_reward_reduce

        if player_name in outcome.private:
            private[player_name] = reduce(inc, outcome.private[player_name])
        else:
            private[player_name] = inc

    return Outcome(joint=outcome.joint, private=frozendict(private))


@dataclass(frozen=True)
class CombinedFromOutcome(Generic[RP, RJ]):
    name: PlayerName

    def __call__(self, outcome: Outcome[RP, RJ]) -> Combined[RJ, RP]:
        # check_isinstance(outcome, Outcome, _self=self)
        if not self.name in outcome.private:
            msg = "Looks like the personal value was not included."
            raise ZValueError(name=self.name, outcome=outcome)
        combined = Combined(joint=outcome.joint.get(self.name, None), personal=outcome.private[self.name])
        return combined


def solve_final_joint(
    sc: SolvingContext[Pr, X, U, Y, RP, RJ, SR], gn: GameNode[Pr, X, U, Y, RP, RJ, SR]
) -> ValueAndActions[Pr, U, RP, RJ]:
    outcome = Outcome(private=frozendict(), joint=gn.joint_final_rewards)
    game_value = sc.game.ps.lift_one(outcome)
    check_set_outcomes(game_value)
    actions = frozendict()
    return ValueAndActions(game_value=game_value, mixed_actions=actions)


def solve_final_personal_both(
    sc: SolvingContext[Pr, X, U, Y, RP, RJ, SR], gn: GameNode[Pr, X, U, Y, RP, RJ, SR]
) -> ValueAndActions[Pr, U, RP, RJ]:
    outcome = Outcome(private=gn.is_final, joint=frozendict())
    game_value = sc.game.ps.lift_one(outcome)
    actions = frozendict()
    return ValueAndActions(game_value=game_value, mixed_actions=actions)
