import itertools
from dataclasses import dataclass, replace
from typing import Dict, Generic, Iterator, Mapping, Set, TypeVar

from frozendict import frozendict
from networkx import simple_cycles
from zuper_commons.types import check_isinstance, ZNotImplementedError, ZValueError

from preferences import PrefConverter, Preference
from . import logger
from .agent import RandomAgent
from .create_joint_game_tree import create_game_tree
from .equilibria import analyze_equilibria, get_all_choices_by_players, get_all_combinations
from .game_def import ASet, Combined, Game, GamePreprocessed, PlayerName, RJ, RP, U, X, Y
from .simulate import simulate1, Simulation
from .structures_solution import (
    GameNode,
    IterationContext,
    Outcome,
    SolvedGameNode,
    SolvingContext,
    ValueAndActions,
)

__all__ = ["solve1"]


def solve_random(gp: GamePreprocessed) -> Simulation:
    policies = {player_name: RandomAgent(player.dynamics) for player_name, player in gp.game.players.items()}
    initial_states = {player_name: list(player.initial)[0] for player_name, player in gp.game.players.items()}
    sim = simulate1(gp.game, policies=policies, initial_states=initial_states, dt=gp.dt)
    logger.info(sim=sim)
    return sim


@dataclass
class Solutions(Generic[X, U, Y, RP, RJ]):
    solved: SolvedGameNode[X, U, Y, RP, RJ]
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
    sc = SolvingContext(
        gp, {}, depth=0, outcome_set_preferences=get_outcome_set_preferences_for_players(gp.game)
    )
    solved = solve(sc, game_tree)
    logger.info("solved", value_actions=solved.va)

    #
    # for player_name in gp.game.players:
    #     personal_tree = get_personal_tree(game_tree, player_name)


    return Solutions(solved=solved, game_tree=game_tree)
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
    game: Game,
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


def all_combinations(actions: Mapping[PlayerName, ASet[U]]) -> Iterator[Mapping[PlayerName, U]]:
    names = list(actions)
    possible = [actions[_] for _ in names]
    for combination in itertools.product(*tuple(possible)):
        c = dict(zip(names, combination))
        yield c


def reduce_future_1(
    gp: GamePreprocessed,
    outcome: Outcome[RP, RJ],
    action: Mapping[PlayerName, U],
    incremental: Mapping[PlayerName, Mapping[U, RP]],
) -> Outcome[RP, RJ]:
    private = {}
    # logger.info(outcome=outcome, action=action, incremental=incremental)
    for name, u in action.items():
        reduce = gp.game.players[name].personal_reward_structure.personal_reward_reduce
        inc = incremental[name][u]
        if name in outcome.private:
            private[name] = reduce(inc, outcome.private[name])
        else:
            private[name] = inc

    return Outcome(joint=outcome.joint, private=frozendict(private))


def reduce_future(
    gp: GamePreprocessed,
    outcomes: ASet[Outcome[RP, RJ]],
    incremental: Mapping[PlayerName, Mapping[U, RP]],
    actions: Mapping[PlayerName, ASet[U]],
) -> ASet[Outcome[RP, RJ]]:
    action: Mapping[PlayerName, U]
    obtained = set()
    for action in all_combinations(actions):
        for outcome in outcomes:
            obtained.add(reduce_future_1(gp, outcome, action, incremental))
    return frozenset(obtained)


def solve(
    sc: SolvingContext[X, U, Y, RP, RJ], gn: GameNode[X, U, Y, RP, RJ]
) -> SolvedGameNode[X, U, Y, RP, RJ]:
    if gn in sc.cache:
        return sc.cache[gn]

    solved: Mapping[Mapping[PlayerName, ASet[U]], ASet[Outcome[RP, RJ]]] = {
        actions: reduce_future(sc.gp, solve(sc, v).va.game_value, gn.incremental, actions)
        for actions, v in gn.outcomes.items()
    }

    va: ValueAndActions[U, RP, RJ]
    # if this is a 1-player node: easy
    if len(gn.states) == 1:
        if len(gn.is_final) == 1:
            va = solve_1_player_final(gn)
        else:
            va = solve_1_player(gn, solved)
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
    solved: Mapping[Mapping[PlayerName, ASet[U]], ASet[Outcome[RP, RJ]]],
) -> ValueAndActions[U, RP, RJ]:
    if not gn.moves:
        msg = "Cannot solve_equilibria if there are no moves "
        raise ZValueError(msg, gn=replace(gn, outcomes={}))
    # logger.info(gn=gn, solved=solved)
    # logger.info(possibilities=list(solved))
    players_active = set(gn.moves)
    preferences: Dict[PlayerName, Preference[ASet[Outcome[RP, RJ]]]]
    preferences = {k: sc.outcome_set_preferences[k] for k in players_active}

    try:
        ea = analyze_equilibria(solved, preferences)
    except ZValueError as e:
        raise ZValueError(moves=gn.moves, states=gn.states) from e
    if len(ea.nondom_nash_equilibria) == 1:
        eq = list(ea.nondom_nash_equilibria)[0]
        game_value = solved[eq]
        return ValueAndActions(game_value=game_value, actions=eq)
    else:
        # multiple non-dominated nash equilibria
        outcomes = set(ea.nondom_nash_equilibria.values())
        # but if there is only one outcome,
        # then that is the game value
        if len(outcomes) == 1:
            game_value = list(outcomes)[0]
            # actions = frozenset(ea.nondom_nash_equilibria)
            actions = get_all_choices_by_players(set(ea.nondom_nash_equilibria))
            return ValueAndActions(game_value=game_value, actions=actions)
        multiple_nash_mix = True  # TODO: make param
        if multiple_nash_mix:
            player2choices = get_all_choices_by_players(set(ea.nondom_nash_equilibria))

            # Assume that we will have any of the combinations
            r = get_all_combinations(player2choices)
            actions: ASet[Mapping[PlayerName, U]] = r
            game_value_: Set[Outcome[RP, RJ]] = set()
            for _ in actions:
                _outcomes = ea.ps[_].outcome
                game_value_.update(_outcomes)
            game_value: ASet[Outcome[RP, RJ]] = frozenset(game_value_)
            return ValueAndActions(game_value=game_value, actions=player2choices)

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

    def __call__(self, a: ASet[Outcome[RP, RJ]]) -> ASet[Combined[RJ, RP]]:
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
    return ValueAndActions(game_value=game_value, actions=actions)


def solve_final_personal_both(gn: GameNode[X, U, Y, RP, RJ]) -> ValueAndActions[U, RP, RJ]:
    game_value = frozenset({Outcome(private=gn.is_final, joint=frozendict())})
    actions = frozendict()
    return ValueAndActions(game_value=game_value, actions=actions)


def solve_1_player_final(gn: GameNode[X, U, Y, RP, RJ]) -> ValueAndActions[U, RP, RJ]:
    p = list(gn.states)[0]
    # logger.info(f"final for {p}")
    game_value = frozenset({Outcome(private=frozendict({p: gn.is_final[p]}), joint=frozendict())})
    actions = frozendict()
    return ValueAndActions(game_value=game_value, actions=actions)


def solve_1_player(
    gn: GameNode[X, U, Y, RP, RJ],
    solved: Mapping[Mapping[PlayerName, ASet[U]], ASet[Outcome[RP, RJ]]],
) -> ValueAndActions[U, RP, RJ]:
    p = list(gn.states)[0]
    action = list(gn.moves[p])[0]
    actions = frozendict({p: frozenset({action})})
    # incremental = gn.incremental[p][action]
    value = solved[actions]
    game_value = frozenset(value)
    return ValueAndActions(game_value=game_value, actions=actions)


