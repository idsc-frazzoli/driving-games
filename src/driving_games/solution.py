import itertools
from collections import defaultdict
from dataclasses import dataclass, replace
from decimal import Decimal as D
from typing import Dict, Generic, Iterator, Mapping, Optional, Set, Tuple, TypeVar

from frozendict import frozendict
from networkx import simple_cycles

from zuper_commons.types import check_isinstance, ZNotImplementedError, ZValueError
from . import logger
from .equilibria import analyze_equilibria, get_all_choices_by_players, get_all_combinations
from .game_def import ASet, Combined, Game, GamePreprocessed, PlayerName, RJ, RP, U, X, Y
from .poset import Preference
from .prefconv import PrefConverter

JointState = Mapping[PlayerName, Optional[X]]
JointAction = Mapping[PlayerName, Optional[U]]


@dataclass
class SolverParams:
    dt: D


def solve1(gp: GamePreprocessed):
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
    logger.info("solved",
                value_actions=solved.va)
    # logger.info(game_tree=game_tree)


@dataclass(frozen=True, unsafe_hash=True, order=True)
class GameNode(Generic[X, U, Y, RP, RJ]):
    states: Mapping[PlayerName, X]
    moves: Mapping[PlayerName, ASet[U]]
    outcomes: "Mapping[Mapping[PlayerName, ASet[U]], GameNode[X, U, Y, RP, RJ]]"

    is_final: Mapping[PlayerName, RP]
    incremental: Mapping[PlayerName, Mapping[U, RP]]

    joint_final_rewards: Mapping[PlayerName, RJ]


@dataclass(frozen=True, unsafe_hash=True, order=True)
class Outcome(Generic[RP, RJ]):
    private: Mapping[PlayerName, RP]
    joint: Mapping[PlayerName, RJ]


@dataclass(frozen=True, unsafe_hash=True, order=True)
class ValueAndActions(Generic[U, RP, RJ]):
    actions: Mapping[PlayerName, ASet[U]]
    game_value: ASet[Outcome[RP, RJ]]

    def __post_init__(self):
        check_isinstance(self.actions, frozendict, me=self)
        for k, v in self.actions.items():
            check_isinstance(v, frozenset, me=self, k=k)
        check_isinstance(self.game_value, frozenset, me=self)
        for o in self.game_value:
            check_isinstance(o, Outcome, me=self)

@dataclass(frozen=True, unsafe_hash=True, order=True)
class SolvedGameNode(Generic[X, U, Y, RP, RJ]):
    gn: GameNode[X, U, Y, RP, RJ]
    solved: "Mapping[Mapping[PlayerName, ASet[U]], SolvedGameNode[X, U, Y, RP, RJ]]"

    va: ValueAndActions[U, RP, RJ]
    # actions: Mapping[PlayerName, ASet[U]]
    # game_value: ASet[Outcome[RP, RJ]]

    def __post_init__(self):
        check_isinstance(self.va, ValueAndActions, me=self)
@dataclass
class SolvingContext(Generic[X, U, Y, RP, RJ]):
    gp: GamePreprocessed[X, U, Y, RP, RJ]
    cache: dict
    depth: int
    outcome_set_preferences: Mapping[PlayerName, Preference[ASet[Outcome]]]


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
            va = solve_1_player( gn, solved)
    else:
        if gn.joint_final_rewards:  # final costs:
            va = solve_final_joint(gn)
        elif set(gn.states) == set(gn.is_final):
            # They both finished
            va = solve_final_personal_both(gn)
        else:
            va = solve_equilibria(sc, gn, solved)

    res = SolvedGameNode(
        gn=gn,
        solved=frozendict(solved),
        va=va
    )
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
            actions = get_all_choices_by_players(ea.nondom_nash_equilibria)
            return ValueAndActions(game_value=game_value, actions=actions)
        multiple_nash_mix = True
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
    raise Exception()


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


#
# class PlayerPreferences(Preference[ASet[Outcome[RP, RJ]]]):
#     p: Preference[Combined[RJ, RP]]
#     pt: Preference[ASet[Combined[RJ, RP]]]
#     name: PlayerName
#     gp: GamePreprocessed
#
#     def __init__(
#         self,
#         gp: GamePreprocessed,
#         name: PlayerName,
#         transformer: Callable[[Preference[M]], Preference[ASet[M]]],
#     ):
#         self.gp = gp
#         self.name = name
#
#         self.p = self.gp.game.players[self.name].preferences
#         self.pt = transformer(self.p)
#
#     def get_type(self):
#         return ASet[self.p.get_type()]
#
#     def __repr__(self):
#         d = {'T': self.get_type(), 'name': self.name, 'pt': self.pt}
#         return 'PlayerPrefs: ' + debug_print(d)
#
#     def transform_to_private(self, a: ASet[Outcome[RP, RJ]]) -> ASet[Combined[RJ, RP]]:
#         check_isinstance(a, frozenset, _self=self)
#         res = set()
#         for s in a:
#             check_isinstance(s, Outcome, _self=self)
#             p = Combined(joint=s.joint.get(self.name, None), personal=s.private[self.name])
#             # logger.info(p=p)
#             res.add(p)
#         return frozenset(res)
#
#     def compare(self, a: ASet[Outcome[RP, RJ]], b: ASet[Outcome[RP, RJ]]) -> ComparisonOutcome:
#         """ <= for the poset """
#         check_isinstance(a, frozenset, _self=self.get_type())
#         check_isinstance(b, frozenset, _self=self.get_type())
#         a_p = self.transform_to_private(a)
#         b_p = self.transform_to_private(b)
#         res = self.pt.compare(a_p, b_p)
#         assert res in COMP_OUTCOMES, (res, self.pt)
#         return res


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


@dataclass
class IterationContext:
    gp: GamePreprocessed[X, U, Y, RP, RJ]
    cache: dict
    depth: int


def create_game_tree(ic: IterationContext, N: JointState) -> GameNode[X, U, Y, RP, RJ]:
    if N in ic.cache:
        return ic.cache[N]
    states = {k: v for k, v in N.items() if v is not None}
    # if ic.depth > 20:
    #     return None
    # get all possible successors
    G = ic.gp.game_graph

    N2: JointState

    moves = defaultdict(set)

    pure_outcomes = {}

    ic2 = replace(ic, depth=ic.depth + 1)
    # noinspection PyArgumentList
    for N_, N2, attrs in G.out_edges(N, data=True):
        joint_action: JointAction = attrs["action"]

        for p, m in joint_action.items():
            if m is not None:
                moves[p].add(m)

        pure_action = frozendict(
            {
                pname: frozenset([action])
                for pname, action in joint_action.items()
                if action is not None
            }
        )
        pure_outcomes[pure_action] = create_game_tree(ic2, N2)

    is_final = {}
    for player_name, player_state in states.items():
        _ = ic.gp.game.players[player_name]
        if _.personal_reward_structure.is_personal_final_state(player_state):
            f = _.personal_reward_structure.personal_final_reward(player_state)
            is_final[player_name] = f

    incremental = defaultdict(dict)
    for k, its_moves in moves.items():
        for move in its_moves:
            pri = ic.gp.game.players[k].personal_reward_structure.personal_reward_incremental
            inc = pri(states[k], move, ic.gp.dt)
            incremental[k][move] = inc

    who_exits = frozenset(ic.gp.game.joint_reward.is_joint_final_state(states))
    joint_final = who_exits
    if joint_final:
        joint_final_rewards = ic.gp.game.joint_reward.joint_reward(states)
    else:
        joint_final_rewards = {}

    moves = {k: frozenset(v) for k, v in moves.items()}

    outcomes = pure_outcomes
    res = GameNode(
        moves=frozendict(moves),
        states=frozendict(states),
        outcomes=frozendict(outcomes),
        incremental=frozendict({k: frozendict(v) for k, v in incremental.items()}),
        joint_final_rewards=frozendict(joint_final_rewards),
        is_final=frozendict(is_final),
    )
    ic.cache[N] = res
    return res
