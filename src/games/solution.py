from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Callable, Dict, Generic, Mapping, Set

from frozendict import frozendict
from networkx import simple_cycles

from possibilities import Poss
from preferences import PrefConverter, Preference
from zuper_commons.types import check_isinstance, ZException, ZNotImplementedError, ZValueError
from . import logger
from .agent import RandomAgent
from .comb_utils import (
    flatten_outcomes,
    get_all_choices_by_players,
    get_all_combinations,
)
from .create_joint_game_tree import create_game_tree
from .equilibria import (
    analyze_equilibria,
    EquilibriaAnalysis,
)
from .game_def import (
    AgentBelief,
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
    U,
    X,
    Y,
)
from .simulate import simulate1, Simulation
from .solution_security import get_security_policies
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
    ValueAndActions,
)

__all__ = ["solve1", "solve_random", "get_outcome_set_preferences_for_players"]


def solve_random(gp: GamePreprocessed[Pr, X, U, Y, RP, RJ]) -> Simulation[Pr, X, U, Y, RP, RJ]:
    ps = gp.game.ps

    policies = {
        player_name: RandomAgent(player.dynamics, ps) for player_name, player in gp.game.players.items()
    }
    initial_states = {
        player_name: list(player.initial.support())[0] for player_name, player in gp.game.players.items()
    }
    sim = simulate1(gp.game, policies=policies, initial_states=initial_states, dt=gp.solver_params.dt, seed=0)
    logger.info(sim=sim)
    return sim


# IState = ASet[JointState]


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

    sims: Dict[str, Simulation] = {}

    cache = {}
    ic = IterationContext(gp, cache, depth=0)
    logger.info("creating game tree")
    game_tree = create_game_tree(ic, initial)
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
            # if x_p2 not in alone_solutions_p2:
            #     raise ZValueError(
            #         x_p2=x_p2, avail=set(alone_solutions_p2), is_it=x_p2 in alone_solutions_p2
            #     )
            policy = alone_solutions_p2[x_p2].policies[p2]
            controllers_others[p2] = AgentFromPolicy(policy)

        tree_ghost = get_ghost_tree(player_name, game_tree, controllers_others)
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
    return Solutions(
        game_solution=game_solution, game_tree=game_tree, solutions_players=solutions_players, sims=sims,
    )
    # logger.info(game_tree=game_tree)


class DoesNotKnowPolicy(ZException):
    pass


class AgentFromPolicy(AgentBelief[Pr, X, U]):
    policy: Mapping[X, Mapping[Poss[JointState, Pr], Poss[U, Pr]]]

    def __init__(self, policy: Mapping[X, Mapping[Poss[JointState, Pr], Poss[U, Pr]]]):
        self.policy = policy

    def get_commands(self, state_self: X, state_others: Poss[JointState, Pr]) -> Poss[U, Pr]:
        if state_self not in self.policy:
            msg = "I do not know the policy for this state"
            raise DoesNotKnowPolicy(
                msg, state_self=state_self, state_others=state_others, states_self_known=set(self.policy),
            )

        lookup = self.policy[state_self]
        if len(lookup) == 1:
            return list(lookup.values())[0]

        if state_others in lookup:
            return lookup[state_others]
        else:
            raise ZNotImplementedError(state_self=state_self, state_others=state_others, lookup=lookup)


def get_ghost_tree(
    player_name: PlayerName,
    game_tree: GameNode[Pr, X, U, Y, RP, RJ],
    controllers: Mapping[PlayerName, AgentBelief[Pr, X, U]],
) -> GameNode[Pr, X, U, Y, RP, RJ]:
    assert len(controllers) >= 1, controllers
    assert player_name not in controllers, (player_name, set(controllers))
    cache: Dict[GameNode[Pr, X, U, Y, RP, RJ], Poss[GameNode[Pr, X, U, Y, RP, RJ], Pr]] = {}
    return replace_others(player_name, game_tree, controllers, cache=cache)


def replace_others(
    dreamer: PlayerName,
    node: GameNode[Pr, X, U, Y, RP, RJ],
    controllers: Mapping[PlayerName, AgentBelief[Pr, X, U]],
    cache: Dict[GameNode[Pr, X, U, Y, RP, RJ], Poss[GameNode[Pr, X, U, Y, RP, RJ], Pr]],
) -> GameNode[Pr, X, U, Y, RP, RJ]:
    if node in cache:
        return cache[node]
    assert dreamer not in controllers
    assert controllers
    # what would they do?
    action_others = {}
    for player_name in node.states:
        # for player_name, controller in controllers.items():
        if player_name in node.is_final:
            continue
        if player_name in node.joint_final_rewards:
            continue
        if player_name == dreamer:
            continue
        state_self = node.states[player_name]
        state_others: JointState = frozendict({k: v for k, v in node.states.items() if k != player_name})
        options = controllers[player_name].get_commands(state_self, state_others)
        if len(options) != 1:
            raise ZNotImplementedError(options=options)
        action_others[player_name] = list(options)[0]

    # find out which actions are compatible
    outcomes = Mapping[JointPureActions, Poss[GameNode[Pr, X, U, Y, RP, RJ], Pr]]
    outcomes = {
        k: replace_others(dreamer, v, controllers, cache)
        for k, v in node.outcomes.items()
        if is_compatible(k, action_others)
    }

    # logger.info(action_others=action_others, original=set(node.outcomes), compatible=set(outcomes))
    moves = get_all_choices_by_players(set(outcomes))
    for player_name in action_others:
        if len(moves[player_name]) != 1:
            raise ZValueError(
                moves=moves, dreamer=dreamer, controllers=list(controllers), orig_moves=node.moves
            )

    # if len(moves) == len(node.moves):
    #     raise ZValueError(moves=moves, dreamer=dreamer, controllers=list(controllers), orig_moves=node.moves)
    ret = GameNode(
        outcomes=frozendict(outcomes),
        states=node.states,
        is_final=node.is_final,
        incremental=node.incremental,
        joint_final_rewards=node.joint_final_rewards,
        moves=moves,
    )
    cache[node] = ret
    return ret


def is_compatible(a: JointPureActions, constraints: JointPureActions) -> bool:
    for k, v in constraints.items():
        if a[k] != v:
            return False
    return True


def get_outcome_set_preferences_for_players(
    game: Game[Pr, X, U, Y, RP, RJ],
) -> Mapping[PlayerName, Preference[SetOfOutcomes]]:
    preferences = {}
    for player_name, player in game.players.items():
        # Comparse Combined[RJ, RP]
        pref0: Preference[Combined[RJ, RP]] = player.preferences
        pref1: Preference[Outcome[RJ, RP]]
        pref1 = PrefConverter(A=Outcome, B=Combined, convert=CombinedFromOutcome(player_name), p0=pref0)
        # compares Aset(Combined[RJ, RP]
        set_preference_aggregator: Callable[[Preference[P]], Preference[Poss[P, Pr]]]
        set_preference_aggregator = player.set_preference_aggregator
        pref2: Preference[SetOfOutcomes] = set_preference_aggregator(pref1)
        # result: Preference[ASet[Outcome[RP, RJ]]]
        preferences[player_name] = pref2
    return preferences


def solve_game(
    gp: GamePreprocessed[Pr, X, U, Y, RP, RJ], gn: GameNode[Pr, X, U, Y, RP, RJ]
) -> GameSolution[Pr, X, U, Y, RP, RJ]:
    outcome_set_preferences = get_outcome_set_preferences_for_players(gp.game)
    sc = SolvingContext(gp, {}, outcome_set_preferences)
    gn_solved = _solve_game(sc, gn)

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

    policies2 = frozendict({k: frozendict(v) for k, v in policies.items()})
    return GameSolution(gn, gn_solved, policies2)


def _solve_game(
    sc: SolvingContext[Pr, X, U, Y, RP, RJ], gn: GameNode[Pr, X, U, Y, RP, RJ]
) -> SolvedGameNode[Pr, X, U, Y, RP, RJ]:
    if gn in sc.cache:
        return sc.cache[gn]

    for pure_actions in gn.outcomes2:
        check_joint_pure_actions(pure_actions)

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
        check_set_outcomes(flattened)
        # Now we need to add the incremental costs
        f = lambda _: add_incremental_cost(gp=sc.gp, incremental_for_player=inc, outcome=_)
        added: SetOfOutcomes = ps.build(flattened, f)

        check_set_outcomes(added)
        solved[pure_actions] = added

    va: ValueAndActions[U, RP, RJ]
    # if this is a 1-player node: easy
    # if False and len(gn.states) == 1:
    #     if len(gn.is_final) == 1:
    #         va = solve_1_player_final(gn)
    #     else:
    #         player_name = list(gn.states)[0]
    #         va = solve_1_player(sc, player_name, gn, solved)
    # else:
    if gn.joint_final_rewards:  # final costs:
        va = solve_final_joint(sc, gn)
    elif set(gn.states) == set(gn.is_final):
        # They both finished
        va = solve_final_personal_both(sc, gn)
    else:
        va = solve_equilibria(sc, gn, solved)

    ret = SolvedGameNode(gn=gn, solved=frozendict(solved_to_node), va=va)
    sc.cache[gn] = ret
    return ret


def add_incremental_cost(
    gp: GamePreprocessed[Pr, X, U, Y, RP, RJ],
    *,
    outcome: Outcome[RP, RJ],
    incremental_for_player: Mapping[PlayerName, RP],
) -> Outcome[RP, RJ]:
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
    gn: GameNode[Pr, X, U, Y, RP, RJ],
    solved: Mapping[JointPureActions, SetOfOutcomes],
) -> ValueAndActions[U, RP, RJ]:
    for pure_action in solved:
        check_joint_pure_actions(pure_action)

    if not gn.moves:
        msg = "Cannot solve_equilibria if there are no moves "
        raise ZValueError(msg, gn=replace(gn, outcomes2={}))
    # logger.info(gn=gn, solved=solved)
    # logger.info(possibilities=list(solved))
    players_active = set(gn.moves)
    preferences: Dict[PlayerName, Preference[SetOfOutcomes]]
    preferences = {k: sc.outcome_set_preferences[k] for k in players_active}

    ea: EquilibriaAnalysis[Pr, X, U, Y, RP, RJ]
    ea = analyze_equilibria(ps=sc.gp.game.ps, moves=gn.moves, solved=solved, preferences=preferences)
    logger.info(ea=ea)
    if len(ea.nondom_nash_equilibria) == 1:
        eq = list(ea.nondom_nash_equilibria)[0]
        check_joint_mixed_actions2(eq)
        # eq_ = mixed_from_pure(eq)
        game_value = ea.nondom_nash_equilibria[eq]
        return ValueAndActions(game_value=game_value, mixed_actions=eq)
    else:
        # multiple non-dominated nash equilibria
        outcomes = set(ea.nondom_nash_equilibria.values())
        # but if there is only one outcome,
        # then that is the game value
        if len(outcomes) == 1:
            game_value = list(outcomes)[0]
            mixed_actions_ = get_all_choices_by_players(set(ea.nondom_nash_equilibria))
            return ValueAndActions(game_value=game_value, mixed_actions=mixed_actions_)

        strategy = sc.gp.solver_params.strategy_multiple_nash
        if strategy == STRATEGY_MIX:
            mixed_actions: JointMixedActions = get_all_choices_by_players(set(ea.nondom_nash_equilibria))

            # Assume that we will have any of the combinations
            set_pure_actions: ASet[JointPureActions]
            set_pure_actions = get_all_combinations(mixed_actions=mixed_actions)
            game_value_: Set[Outcome[RP, RJ]] = set()
            for pure_action in set_pure_actions:
                _outcomes = ea.ps[pure_action].outcome
                game_value_.update(_outcomes)
            game_value__: SetOfOutcomes = frozenset(game_value_)
            return ValueAndActions(game_value=game_value__, mixed_actions=mixed_actions)

        elif strategy == STRATEGY_SECURITY:
            security_policies: JointMixedActions2 = get_security_policies(
                sc.gp.game.ps, solved, gn.moves, sc.outcome_set_preferences
            )
            set_pure_actions = get_all_combinations(mixed_actions=security_policies)
            set_outcomes: SetOfOutcomes = flatten_outcomes(solved, set_pure_actions)
            return ValueAndActions(game_value=set_outcomes, mixed_actions=security_policies)
        elif strategy == STRATEGY_BAIL:
            msg = "Multiple Nash Equilibria"
            raise ZNotImplementedError(msg, ea=ea)
        else:
            assert False, strategy


class TransformToPrivate0(Generic[Pr, X, U, Y, RP, RJ]):
    gp: GamePreprocessed[Pr, X, U, Y, RP, RJ]
    name: PlayerName

    def __init__(self, gp: GamePreprocessed[Pr, X, U, Y, RP, RJ], name: PlayerName):
        self.gp = gp
        self.name = name

    def __call__(self, a: SetOfOutcomes) -> Poss[Combined[RJ, RP], Pr]:
        check_isinstance(a, frozenset, _self=self)

        def f(s: Outcome[RP, RJ]) -> Combined[RJ, RP]:
            return Combined(joint=s.joint.get(self.name, None), personal=s.private[self.name])

        return self.gp.game.ps.build(a, f)


@dataclass
class CombinedFromOutcome(Generic[RP, RJ]):
    name: PlayerName

    def __call__(self, outcome: Outcome[RP, RJ]) -> Combined[RJ, RP]:
        check_isinstance(outcome, Outcome, _self=self)
        combined = Combined(joint=outcome.joint.get(self.name, None), personal=outcome.private[self.name])
        return combined


def solve_final_joint(
    sc: SolvingContext[Pr, X, U, Y, RP, RJ], gn: GameNode[Pr, X, U, Y, RP, RJ]
) -> ValueAndActions[U, RP, RJ]:
    outcome = Outcome(private=frozendict(), joint=gn.joint_final_rewards)
    game_value = sc.gp.game.ps.lift_one(outcome)
    check_set_outcomes(game_value)
    actions = frozendict()
    return ValueAndActions(game_value=game_value, mixed_actions=actions)


def solve_final_personal_both(
    sc: SolvingContext[Pr, X, U, Y, RP, RJ], gn: GameNode[Pr, X, U, Y, RP, RJ]
) -> ValueAndActions[U, RP, RJ]:
    outcome = Outcome(private=gn.is_final, joint=frozendict())
    game_value = sc.gp.game.ps.lift_one(outcome)
    actions = frozendict()
    return ValueAndActions(game_value=game_value, mixed_actions=actions)


#
# def solve_1_player_final(gn: GameNode[Pr, X, U, Y, RP, RJ]) -> ValueAndActions[U, RP, RJ]:
#     p = list(gn.states)[0]
#     # logger.info(f"final for {p}")
#     game_value = frozenset({Outcome(private=frozendict({p: gn.is_final[p]}), joint=frozendict())})
#     actions = frozendict()
#     return ValueAndActions(game_value=game_value, mixed_actions=actions)

#
# def solve_1_player(
#     sc: SolvingContext[Pr, X, U, Y, RP, RJ],
#     player_name: PlayerName,
#     gn: GameNode[Pr, X, U, Y, RP, RJ],
#     solved: Mapping[JointPureActions, SetOfOutcomes],
# ) -> ValueAndActions[U, RP, RJ]:
#     for pure_action in solved:
#         check_joint_pure_actions(pure_action)
#
#     pref = sc.outcome_set_preferences[player_name]
#     nondominated = remove_dominated(solved, pref)
#
#     all_actions = frozenset({_[player_name] for _ in nondominated})
#     actions = frozendict({player_name: all_actions})
#
#     value = set()
#     for action, out in nondominated.items():
#         value.update(out)
#
#     game_value = frozenset(value)
#
#     return ValueAndActions(game_value=game_value, mixed_actions=actions)
