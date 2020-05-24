from collections import defaultdict
from dataclasses import dataclass, replace
from decimal import Decimal as D
from typing import Callable, Dict, FrozenSet, Generic, Mapping

from frozendict import frozendict
from networkx import simple_cycles

from possibilities import Poss
from preferences import PrefConverter, Preference
from zuper_commons.types import check_isinstance, ZNotImplementedError, ZValueError
from . import logger
from .agent_from_policy import AgentFromPolicy
from .create_joint_game_tree import create_game_tree
from .equilibria import (
    analyze_equilibria,
    EquilibriaAnalysis,
)
from .game_def import (
    check_joint_mixed_actions2,
    Combined,
    Game,
    JointMixedActions2,
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
from .solution_security import get_mixed2, get_security_policies
from .structures_solution import (
    check_joint_pure_actions,
    check_set_outcomes,
    GameNode,
    GamePreprocessed,
    GameSolution,
    IterationContext,
    Solutions,
    SolutionsPlayer,
    SolvedGameNode,
    SolvingContext,
    STRATEGY_BAIL,
    STRATEGY_MIX,
    STRATEGY_SECURITY,
    UsedResources,
    ValueAndActions,
)

__all__ = ["solve1", "get_outcome_set_preferences_for_players"]


def solve1(gp: GamePreprocessed[Pr, X, U, Y, RP, RJ]) -> Solutions[Pr, X, U, Y, RP, RJ]:
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

    cache = {}
    ic = IterationContext(gp, cache, depth=0)
    logger.info("creating game tree")
    game_tree = create_game_tree(ic, initial)
    logger.info(f"nodes: {len(ic.cache)}")

    solutions_players: Dict[PlayerName, SolutionsPlayer] = {}
    initial_state = game_tree.states
    alone_solutions: Dict[PlayerName, Dict[X, GameSolution]] = {}
    for player_name, pp in gp.players_pre.items():
        alone_solutions[player_name] = {}
        for x0, personal_tree in pp.alone_tree.items():
            solved_x0 = solve_game(gp, personal_tree)
            alone_solutions[player_name][x0] = solved_x0
            logger.info(
                f"Solution for {player_name} alone",
                game_value=solved_x0.gn_solved.va.game_value,
                # policy=solved_x0.policies
            )

    for player_name, pp in gp.players_pre.items():
        # use other solutions
        # logger.info("looking for ghost solutions")
        controllers_others = {}
        for p2 in gp.players_pre:
            if p2 == player_name:
                continue
            x_p2 = initial_state[p2]
            alone_solutions_p2 = alone_solutions[p2]

            policy = alone_solutions_p2[x_p2].policies[p2]
            controllers_others[p2] = AgentFromPolicy(policy)

        tree_ghost = get_ghost_tree(gp, player_name, game_tree, controllers_others)
        # logger.info(
        #     "first node of tree ghost",
        #     tree_ghost=replace(tree_ghost, outcomes=frozendict()),
        #     outcomes=set(tree_ghost.outcomes),
        # )
        solution_ghost = solve_game(gp, tree_ghost)
        logger.info(
            f"Stackelberg solution when {player_name} is a follower",
            game_value=solution_ghost.gn_solved.va.game_value,
            # policy=solution_ghost.policies,
        )
        controllers = dict(controllers_others)
        controllers[player_name] = AgentFromPolicy(solution_ghost.policies[player_name])
        sim_ = simulate1(
            gp.game, policies=controllers, initial_states=initial_state, dt=gp.solver_params.dt, seed=0,
        )
        sims[f"{player_name}-follows"] = sim_

    logger.info("solving game tree")
    game_solution = solve_game(gp, game_tree)
    controllers0 = {}
    for player_name, pp in gp.players_pre.items():
        policy = game_solution.policies[player_name]
        controllers0[player_name] = AgentFromPolicy(policy)
    logger.info(
        f"Value of joint solution",
        game_value=game_solution.gn_solved.va.game_value,
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


def solve_game(
    gp: GamePreprocessed[Pr, X, U, Y, RP, RJ], gn: GameNode[Pr, X, U, Y, RP, RJ, SR]
) -> GameSolution[Pr, X, U, Y, RP, RJ]:
    outcome_set_preferences = get_outcome_set_preferences_for_players(gp.game)
    sc = SolvingContext(gp, outcome_set_preferences, {}, set())
    gn_solved = _solve_game(sc, gn)

    states_to_solution: Dict[JointState, SolvedGameNode] = {}
    for js, sgn in sc.cache.items():
        states_to_solution[js.states] = sgn

    policies: Dict[PlayerName, Dict[X, Dict[Poss[JointState, Pr], Poss[U, Pr]]]]
    ps = gp.game.ps
    policies = defaultdict(lambda: defaultdict(dict))
    for g0, s0 in sc.cache.items():
        state = g0.states
        for player_name, player_state in state.items():

            if player_name in s0.va.mixed_actions:
                policy_for_this_state = policies[player_name][player_state]
                other_states = frozendict({k: v for k, v in state.items() if k != player_name})
                iset = ps.lift_one(other_states)  # frozenset({other_states})
                policy_for_this_state[iset] = s0.va.mixed_actions[player_name]

    # policies2 = frozendict(valmap(policies, )
    policies2 = frozendict({k: fr(v) for k, v in policies.items()})
    return GameSolution(
        gn=gn, gn_solved=gn_solved, policies=policies2, states_to_solution=frozendict(states_to_solution)
    )


def fr(d):
    return frozendict({k: frozendict(v) for k, v in d.items()})


def _solve_game(
    sc: SolvingContext[Pr, X, U, Y, RP, RJ], gn: GameNode[Pr, X, U, Y, RP, RJ, SR]
) -> SolvedGameNode[Pr, X, U, Y, RP, RJ, SR]:
    if gn in sc.cache:
        return sc.cache[gn]
    if gn.states in sc.processing:
        msg = "Loop found"
        raise ZValueError(msg, states=gn.states)

    sc.processing.add(gn.states)

    # logger.debug(gn_states=gn.states, processing=len(sc.processing),
    #              processed=len(sc.cache))
    # for pure_actions in gn.outcomes2:
    #     check_joint_pure_actions(pure_actions)

    ps = sc.gp.game.ps
    # what happens for each action?
    pure_actions: JointPureActions
    solved: Dict[JointPureActions, SetOfOutcomes] = {}
    solved_to_node: Dict[JointPureActions, Poss[SolvedGameNode[Pr, X, U, U, RP, RJ], Pr]] = {}

    for pure_actions in gn.outcomes2:
        # Incremental costs incurred if choosing this action
        inc: Dict[PlayerName, RP]
        inc = {p: gn.incremental[p][u] for p, u in pure_actions.items()}
        # if we choose these actions, then these are the game nodes
        # we could go in
        next_nodes: Poss[GameNode[Pr, X, U, U, RP, RJ], Pr] = gn.outcomes2[pure_actions]
        # and these are the solved nodes (recursive step here)
        next_nodes_solutions: Poss[SolvedGameNode[Pr, X, U, U, RP, RJ], Pr]
        next_nodes_solutions = ps.build(next_nodes, lambda _: _solve_game(sc, _))
        solved_to_node[pure_actions] = next_nodes_solutions
        # For each, we find the solution
        next_outcomes_solutions: Poss[SetOfOutcomes, Pr]
        next_outcomes_solutions = ps.build(next_nodes_solutions, lambda _: _.va.game_value)
        # and now we flatten
        flattened: SetOfOutcomes = ps.flatten(next_outcomes_solutions)
        # check_set_outcomes(flattened)
        # Now we need to add the incremental costs
        f = lambda _: add_incremental_cost(gp=sc.gp, incremental_for_player=inc, outcome=_)
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
    sc.cache[gn] = ret
    sc.processing.remove(gn.states)

    n = len(sc.cache)
    if n % 30 == -1:
        logger.info(
            states=gn.states, value=va.game_value, processing=len(sc.processing), solved=len(sc.cache)
        )
        # logger.info(f"nsolved: {n}")  # , game_value=va.game_value)
    return ret


def add_incremental_cost(
    gp: GamePreprocessed[Pr, X, U, Y, RP, RJ],
    *,
    outcome: Outcome[RP, RJ],
    incremental_for_player: Mapping[PlayerName, RP],
) -> Outcome[RP, RJ]:
    check_isinstance(outcome, Outcome)
    private = {}
    # logger.info(outcome=outcome, action=action, incremental=incremental)
    u: U
    for player_name, inc in incremental_for_player.items():
        reduce = gp.game.players[player_name].personal_reward_structure.personal_reward_reduce

        if player_name in outcome.private:
            private[player_name] = reduce(inc, outcome.private[player_name])
        else:
            private[player_name] = inc

    return Outcome(joint=outcome.joint, private=frozendict(private))


def solve_equilibria(
    sc: SolvingContext[Pr, X, U, Y, RP, RJ],
    gn: GameNode[Pr, X, U, Y, RP, RJ, SR],
    solved: Mapping[JointPureActions, SetOfOutcomes],
) -> ValueAndActions[Pr, U, RP, RJ]:
    ps = sc.gp.game.ps
    # for pure_action in solved:
    #     check_joint_pure_actions(pure_action)

    if not gn.moves:
        msg = "Cannot solve_equilibria if there are no moves "
        raise ZValueError(msg, gn=replace(gn, outcomes2=frozendict()))
    # logger.info(gn=gn, solved=solved)
    # logger.info(possibilities=list(solved))
    players_active = set(gn.moves)
    preferences: Dict[PlayerName, Preference[SetOfOutcomes]]
    preferences = {k: sc.outcome_set_preferences[k] for k in players_active}

    ea: EquilibriaAnalysis[Pr, X, U, Y, RP, RJ]
    ea = analyze_equilibria(ps=sc.gp.game.ps, moves=gn.moves, solved=solved, preferences=preferences)
    # logger.info(ea=ea)
    if len(ea.nondom_nash_equilibria) == 1:
        eq = list(ea.nondom_nash_equilibria)[0]
        check_joint_mixed_actions2(eq)
        # eq_ = mixed_from_pure(eq)
        game_value = ea.nondom_nash_equilibria[eq]
        return ValueAndActions(game_value=game_value, mixed_actions=eq)
    else:
        # multiple nondominated, but same outcome

        # multiple non-dominated nash equilibria
        outcomes = set(ea.nondom_nash_equilibria.values())

        strategy = sc.gp.solver_params.strategy_multiple_nash
        if strategy == STRATEGY_MIX:
            # XXX: Not really sure this makes sense when there are probabilities
            profile: Dict[PlayerName, Poss[U, Pr]] = {}
            for player_name in players_active:
                # find all the mixed strategies he would play at equilibria
                res = set()
                for _ in ea.nondom_nash_equilibria:
                    res.add(_[player_name])
                strategy = ps.flatten(ps.lift_many(res))
                # check_poss(strategy)
                profile[player_name] = strategy

            def f(y: JointPureActions) -> JointPureActions:
                return frozendict(y)

            dist: Poss[JointPureActions, Pr] = ps.build_multiple(a=profile, f=f)

            game_value1: SetOfOutcomes
            game_value1 = ps.flatten(ps.build(dist, solved.__getitem__))
            # logger.info(dist=dist, game_value1=game_value1)
            return ValueAndActions(game_value=game_value1, mixed_actions=frozendict(profile))
        # Anything can happen
        elif strategy == STRATEGY_SECURITY:
            ps = sc.gp.game.ps
            security_policies: JointMixedActions2
            security_policies = get_security_policies(ps, solved, sc.outcome_set_preferences, ea)
            check_joint_mixed_actions2(security_policies)
            dist: Poss[JointPureActions, Pr]
            dist = get_mixed2(ps, security_policies)
            # logger.info(dist=dist)
            for _ in dist.support():
                check_joint_pure_actions(_)

            set_outcomes: SetOfOutcomes = ps.flatten(ps.build(dist, solved.__getitem__))
            return ValueAndActions(game_value=set_outcomes, mixed_actions=security_policies)
        elif strategy == STRATEGY_BAIL:
            msg = "Multiple Nash Equilibria"
            raise ZNotImplementedError(msg, ea=ea)
        else:
            assert False, strategy


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
    sc: SolvingContext[Pr, X, U, Y, RP, RJ], gn: GameNode[Pr, X, U, Y, RP, RJ, SR]
) -> ValueAndActions[Pr, U, RP, RJ]:
    outcome = Outcome(private=frozendict(), joint=gn.joint_final_rewards)
    game_value = sc.gp.game.ps.lift_one(outcome)
    check_set_outcomes(game_value)
    actions = frozendict()
    return ValueAndActions(game_value=game_value, mixed_actions=actions)


def solve_final_personal_both(
    sc: SolvingContext[Pr, X, U, Y, RP, RJ], gn: GameNode[Pr, X, U, Y, RP, RJ, SR]
) -> ValueAndActions[Pr, U, RP, RJ]:
    outcome = Outcome(private=gn.is_final, joint=frozendict())
    game_value = sc.gp.game.ps.lift_one(outcome)
    actions = frozendict()
    return ValueAndActions(game_value=game_value, mixed_actions=actions)
