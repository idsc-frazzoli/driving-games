from dataclasses import dataclass, replace
from typing import Dict, Generic, Mapping, Set, TypeVar

from frozendict import frozendict
from networkx import simple_cycles
from zuper_commons.types import check_isinstance, ZNotImplementedError, ZValueError
from collections import defaultdict
from preferences import PrefConverter, Preference, remove_dominated
from . import GamePreprocessed, get_all_choices_by_players, get_all_combinations, logger
from .agent import RandomAgent
from .comb_utils import all_pure_actions, mixed_from_pure
from .create_joint_game_tree import create_game_tree
from .equilibria import (
    analyze_equilibria,
    EquilibriaAnalysis,
)
from .game_def import (
    ASet,
    Combined,
    Game,
    JointMixedActions,
    JointPureActions,
    PlayerName,
    RJ,
    RP,
    U,
    X,
    Y,
)
from .simulate import simulate1, Simulation
from .structures_solution import (
    check_joint_mixed_actions,
    check_joint_pure_actions,
    check_set_outcomes,
    GameNode,
    IterationContext,
    Outcome,
    SetOfOutcomes,
    SolvedGameNode,
    SolvingContext,
    ValueAndActions,
)

__all__ = ["solve1"]


def solve_random(gp: GamePreprocessed[X, U, Y, RP, RJ]) -> Simulation[X, U, Y, RP, RJ]:
    policies = {
        player_name: RandomAgent(player.dynamics) for player_name, player in gp.game.players.items()
    }
    initial_states = {
        player_name: list(player.initial)[0] for player_name, player in gp.game.players.items()
    }
    sim = simulate1(gp.game, policies=policies, initial_states=initial_states, dt=gp.dt)
    logger.info(sim=sim)
    return sim


@dataclass
class GameSolution(Generic[X, U, Y, RP, RJ]):
    gn: GameNode[X, U, Y, RP, RJ]
    gn_solved: SolvedGameNode[X, U, Y, RP, RJ]

    IState = ASet[Mapping[PlayerName, X]]
    policies: Mapping[PlayerName, Mapping[X, Mapping[IState, ASet[U]]]]

    def __post_init__(self):
        if False:
            for player_name, player_policy in self.policies.items():

                check_isinstance(player_policy, frozendict)
                for own_state, state_policy in player_policy.items():
                    check_isinstance(state_policy, frozendict)
                    for istate, us in state_policy.items():
                        check_isinstance(us, frozenset)


@dataclass
class Solutions(Generic[X, U, Y, RP, RJ]):
    game_solution: GameSolution[X, U, Y, RP, RJ]
    game_tree: GameNode[X, U, Y, RP, RJ]


def solve1(gp: GamePreprocessed[X, U, Y, RP, RJ]) -> Solutions[X, U, Y, RP, RJ]:
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

    cache = {}
    ic = IterationContext(gp, cache, depth=0)
    logger.info("creating game tree")
    game_tree = create_game_tree(ic, initial)
    logger.info("solving game tree")

    game_solution = solve_game(gp, game_tree)
    logger.info("solved", value_actions=game_solution.gn_solved.va, policy=game_solution.policies)
    a = 2
    #
    for player_name, pp in gp.players_pre.items():
        for x0, personal_tree in pp.alone_tree.items():
            solved_x0 = solve_game(gp, personal_tree)
            logger.info(
                "alone solution", value_actions=solved_x0.gn_solved.va, policy=solved_x0.policies
            )

    return Solutions(game_solution=game_solution, game_tree=game_tree)
    # logger.info(game_tree=game_tree)


#
# @dataclass
# class PersonalContext:
#     cache: dict
# def get_personal_tree(node: GameNode, player_name: PlayerName) -> GameNode:
#     """ Removes the others from the tree """
#     context =  PersonalContext({})
#     return get_personal_tree_(context, node, player_name)
#
#
# def get_personal_tree_(pc: PersonalContext, node: GameNode, player_name: PlayerName) -> GameNode:
#     if node.states in pc.cache:
#         return pc.cache[node.states]
#
#     if node.is_final.get(player_name, None):
#         outcomes = {}
#     else:
#         outcomes = {}
#         for actions, outcome in node.outcomes.items():
#             outcomes[only_keep(actions, player_name)] = get_personal_tree_(pc, outcome, player_name)
#
#     is_final = only_keep(node.is_final, player_name)
#     moves = only_keep(node.moves, player_name)
#     states = only_keep(node.states, player_name)
#     incremental= only_keep(node.incremental, player_name)
#     joint_final_rewards = frozendict()
#     res = GameNode(states=states, moves=moves, outcomes=outcomes, is_final=is_final,
#                     incremental=incremental, joint_final_rewards=joint_final_rewards)
#     pc.cache[node.states] = res
#     return res
#
# K = TypeVar('K')
# V = TypeVar('V')
# def only_keep(d: Mapping[K, V], one: K) -> Mapping[K, V]:
#     return frozendict({k: v for k,v in d.items() if k == one})
#


def get_outcome_set_preferences_for_players(
    game: Game[X, U, Y, RP, RJ],
) -> Mapping[PlayerName, Preference[ASet[Outcome]]]:
    preferences = {}
    for player_name, player in game.players.items():
        # Comparse Combined[RJ, RP]
        pref0: Preference[Combined] = player.preferences
        pref1: Preference[Outcome] = PrefConverter(
            A=Outcome, B=Combined, convert=CombinedFromOutcome(player_name), p0=pref0
        )
        # compares Aset(Combined[RJ, RP]
        pref2: Preference[ASet[Outcome]] = player.set_preference_aggregator(pref1)
        # result: Preference[ASet[Outcome[RP, RJ]]]
        preferences[player_name] = pref2
    return preferences


def reduce_future_1(
    gp: GamePreprocessed[X, U, Y, RP, RJ],
    *,
    outcome: Outcome[RP, RJ],
    pure_actions: JointPureActions,
    incremental: Mapping[PlayerName, Mapping[U, RP]],
) -> Outcome[RP, RJ]:
    private = {}
    # logger.info(outcome=outcome, action=action, incremental=incremental)
    u: U
    for name, u in pure_actions.items():
        reduce = gp.game.players[name].personal_reward_structure.personal_reward_reduce
        inc = incremental[name][u]
        if name in outcome.private:
            private[name] = reduce(inc, outcome.private[name])
        else:
            private[name] = inc

    return Outcome(joint=outcome.joint, private=frozendict(private))


def reduce_future(
    gp: GamePreprocessed[X, U, Y, RP, RJ],
    *,
    mixed_outcomes: SetOfOutcomes,
    # incremental cost associated
    incremental: Mapping[PlayerName, Mapping[U, RP]],
    mixed_actions: JointMixedActions,
) -> SetOfOutcomes:
    """


    """
    check_set_outcomes(mixed_outcomes)
    check_joint_mixed_actions(mixed_actions)
    # action: Mapping[PlayerName, U]
    obtained = set()
    pure_actions: JointPureActions
    for pure_actions in all_pure_actions(mixed_actions=mixed_actions):
        check_joint_pure_actions(pure_actions)
        for outcome in mixed_outcomes:
            more = reduce_future_1(
                gp, outcome=outcome, pure_actions=pure_actions, incremental=incremental
            )
            obtained.add(more)
    ret = frozenset(obtained)
    check_set_outcomes(mixed_outcomes)
    return ret


def solve_game(
    gp: GamePreprocessed[X, U, Y, RP, RJ], gn: GameNode[X, U, Y, RP, RJ]
) -> GameSolution[X, U, Y, RP, RJ]:
    sc = SolvingContext(
        gp, {}, depth=0, outcome_set_preferences=get_outcome_set_preferences_for_players(gp.game)
    )
    gn_solved = _solve_game(sc, gn)

    policies: Dict[PlayerName, Dict[X, Dict[ASet[Mapping[PlayerName, X]], ASet[U]]]]

    policies = defaultdict(lambda: defaultdict(dict))
    for g0, s0 in sc.cache.items():
        state = g0.states
        for player_name, player_state in state.items():

            if player_name in s0.va.mixed_actions:
                policy_for_this_state = policies[player_name][player_state]
                other_states = frozendict({k: v for k, v in state.items() if k != player_name})
                iset = frozenset({other_states})
                policy_for_this_state[iset] = s0.va.mixed_actions[player_name]

    policies2 = frozendict({k: frozendict(v) for k, v in policies.items()})
    return GameSolution(gn, gn_solved, policies2)


def _solve_game(
    sc: SolvingContext[X, U, Y, RP, RJ], gn: GameNode[X, U, Y, RP, RJ]
) -> SolvedGameNode[X, U, Y, RP, RJ]:
    if gn in sc.cache:
        return sc.cache[gn]

    for pure_actions in gn.outcomes:
        check_joint_pure_actions(pure_actions)

    solved: Dict[JointPureActions, SetOfOutcomes] = {}
    for pure_actions, v in gn.outcomes.items():
        check_joint_pure_actions(pure_actions)

        mixed_actions = mixed_from_pure(pure_actions)
        mixed_outcomes = _solve_game(sc, v).va.game_value
        res = reduce_future(
            sc.gp,
            mixed_outcomes=mixed_outcomes,
            incremental=gn.incremental,
            mixed_actions=mixed_actions,
        )
        check_set_outcomes(res)
        solved[pure_actions] = res

    va: ValueAndActions[U, RP, RJ]
    # if this is a 1-player node: easy
    if len(gn.states) == 1:
        if len(gn.is_final) == 1:
            va = solve_1_player_final(gn)
        else:
            player_name = list(gn.states)[0]
            va = solve_1_player(sc, player_name, gn, solved)
    else:
        if gn.joint_final_rewards:  # final costs:
            va = solve_final_joint(gn)
        elif set(gn.states) == set(gn.is_final):
            # They both finished
            va = solve_final_personal_both(gn)
        else:
            va = solve_equilibria(sc, gn, solved)

    res = SolvedGameNode(gn=gn, solved=frozendict(solved), va=va)
    sc.cache[gn] = res
    return res


def solve_equilibria(
    sc: SolvingContext[X, U, Y, RP, RJ],
    gn: GameNode[X, U, Y, RP, RJ],
    solved: Mapping[JointPureActions, SetOfOutcomes],
) -> ValueAndActions[U, RP, RJ]:
    for pure_action in solved:
        check_joint_pure_actions(pure_action)

    if not gn.moves:
        msg = "Cannot solve_equilibria if there are no moves "
        raise ZValueError(msg, gn=replace(gn, outcomes={}))
    # logger.info(gn=gn, solved=solved)
    # logger.info(possibilities=list(solved))
    players_active = set(gn.moves)
    preferences: Dict[PlayerName, Preference[SetOfOutcomes]]
    preferences = {k: sc.outcome_set_preferences[k] for k in players_active}

    # try:
    ea: EquilibriaAnalysis[U, SetOfOutcomes]
    ea = analyze_equilibria(solved, preferences)
    # except ZValueError as e:
    #     raise ZValueError(moves=gn.moves, states=gn.states) from e
    if len(ea.nondom_nash_equilibria) == 1:
        eq = list(ea.nondom_nash_equilibria)[0]
        check_joint_pure_actions(eq)
        eq_ = mixed_from_pure(eq)
        game_value = solved[eq]
        return ValueAndActions(game_value=game_value, mixed_actions=eq_)
    else:
        # multiple non-dominated nash equilibria
        outcomes = set(ea.nondom_nash_equilibria.values())
        # but if there is only one outcome,
        # then that is the game value
        if len(outcomes) == 1:
            game_value = list(outcomes)[0]
            mixed_actions_ = get_all_choices_by_players(set(ea.nondom_nash_equilibria))
            return ValueAndActions(game_value=game_value, mixed_actions=mixed_actions_)
        multiple_nash_mix = True  # TODO: make param
        if multiple_nash_mix:
            player2choices = get_all_choices_by_players(set(ea.nondom_nash_equilibria))

            # Assume that we will have any of the combinations
            set_pure_actions: ASet[JointPureActions] = get_all_combinations(
                mixed_actions=player2choices
            )
            game_value_: Set[Outcome[RP, RJ]] = set()
            for pure_action in set_pure_actions:
                _outcomes = ea.ps[pure_action].outcome
                game_value_.update(_outcomes)
            game_value: SetOfOutcomes = frozenset(game_value_)
            return ValueAndActions(game_value=game_value, mixed_actions=player2choices)

        else:

            msg = "Multiple Nash Equilibria"
            raise ZNotImplementedError(msg, ea=ea)


M = TypeVar("M")


class TransformToPrivate0(Generic[X, U, Y, RP, RJ]):
    gp: GamePreprocessed[X, U, Y, RP, RJ]
    name: PlayerName

    def __init__(self, gp: GamePreprocessed[X, U, Y, RP, RJ], name: PlayerName):
        self.gp = gp
        self.name = name

    def __call__(self, a: SetOfOutcomes) -> ASet[Combined[RJ, RP]]:
        check_isinstance(a, frozenset, _self=self)
        res = set()
        for s in a:
            check_isinstance(s, Outcome, _self=self)
            p = Combined(joint=s.joint.get(self.name, None), personal=s.private[self.name])
            # logger.info(p=p)
            res.add(p)
        return frozenset(res)


@dataclass
class CombinedFromOutcome(Generic[RP, RJ]):
    name: PlayerName

    def __call__(self, outcome: Outcome[RP, RJ]) -> Combined[RJ, RP]:
        check_isinstance(outcome, Outcome, _self=self)
        combined = Combined(
            joint=outcome.joint.get(self.name, None), personal=outcome.private[self.name]
        )
        return combined


def solve_final_joint(gn: GameNode[X, U, Y, RP, RJ]) -> ValueAndActions[U, RP, RJ]:
    game_value = frozenset({Outcome(private=frozendict(), joint=gn.joint_final_rewards)})
    actions = frozendict()
    return ValueAndActions(game_value=game_value, mixed_actions=actions)


def solve_final_personal_both(gn: GameNode[X, U, Y, RP, RJ]) -> ValueAndActions[U, RP, RJ]:
    game_value = frozenset({Outcome(private=gn.is_final, joint=frozendict())})
    actions = frozendict()
    return ValueAndActions(game_value=game_value, mixed_actions=actions)


def solve_1_player_final(gn: GameNode[X, U, Y, RP, RJ]) -> ValueAndActions[U, RP, RJ]:
    p = list(gn.states)[0]
    # logger.info(f"final for {p}")
    game_value = frozenset({Outcome(private=frozendict({p: gn.is_final[p]}), joint=frozendict())})
    actions = frozendict()
    return ValueAndActions(game_value=game_value, mixed_actions=actions)


def solve_1_player(
    sc: SolvingContext[X, U, Y, RP, RJ],
    player_name: PlayerName,
    gn: GameNode[X, U, Y, RP, RJ],
    solved: Mapping[JointPureActions, SetOfOutcomes],
) -> ValueAndActions[U, RP, RJ]:
    for pure_action in solved:
        check_joint_pure_actions(pure_action)
    # p = list(gn.states)[0]
    # action = list(gn.moves[p])[0]

    pref = sc.outcome_set_preferences[player_name]
    nondominated = remove_dominated(solved, pref)

    all_actions = frozenset({_[player_name] for _ in nondominated})
    actions = frozendict({player_name: all_actions})
    # incremental = gn.incremental[p][action]
    # if actions not in solved:
    #     raise ZValueError(actions=actions, solved=solved)
    value = set()
    for action, out in nondominated.items():
        value.update(out)
    # value = solved[actions]
    game_value = frozenset(value)

    return ValueAndActions(game_value=game_value, mixed_actions=actions)
