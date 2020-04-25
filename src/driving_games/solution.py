import itertools
from collections import defaultdict
from dataclasses import dataclass, replace
from decimal import Decimal as D
from typing import Callable, Dict, FrozenSet, Generic, Iterator, Mapping, Optional, Tuple, TypeVar

from frozendict import frozendict
from networkx import simple_cycles

from zuper_commons.types import check_isinstance, ZNotImplementedError, ZValueError
from zuper_typing import debug_print
from . import logger
from .equilibria import analyze_equilibria
from .game_def import Combined, GamePreprocessed, PlayerName
from .poset import COMP_OUTCOMES, ComparisonOutcome, Preference
from .poset_sets import SetPreference1

ASet = FrozenSet


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
    sc = SolvingContext(gp, {}, depth=0)
    solved = solve(sc, game_tree)
    logger.info("done")
    # logger.info(game_tree=game_tree)


from .game_def import X, U, RJ, RP, Y, ASet

JointState = Mapping[PlayerName, Optional[X]]
JointAction = Mapping[PlayerName, Optional[U]]


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
class SolvedGameNode(Generic[X, U, Y, RP, RJ]):
    gn: GameNode[X, U, Y, RP, RJ]
    solved: "Mapping[Mapping[PlayerName, ASet[U]], SolvedGameNode[X, U, Y, RP, RJ]]"

    actions: Mapping[PlayerName, ASet[U]]
    game_value: ASet[Outcome[RP, RJ]]


@dataclass
class SolvingContext(Generic[X, U, Y, RP, RJ]):
    gp: GamePreprocessed[X, U, Y, RP, RJ]
    cache: dict
    depth: int


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
        actions: reduce_future(sc.gp, solve(sc, v).game_value, gn.incremental, actions)
        for actions, v in gn.outcomes.items()
    }

    # if this is a 1-player node: easy
    if len(gn.states) == 1:
        if len(gn.is_final) == 1:
            game_value, actions = solve_1_player_final(gn)
        else:
            game_value, actions = solve_1_player(sc, gn, solved)
    else:
        if gn.joint_final_rewards:  # final costs:
            game_value, actions = solve_final_joint(gn)
        elif set(gn.states) == set(gn.is_final):
            # They both finished
            game_value, actions = solve_final_personal_both(gn)
        else:
            game_value, actions = solve_equilibria(sc, gn, solved)

    res = SolvedGameNode(
        gn=gn,
        solved=frozendict(solved),
        game_value=frozenset(game_value),
        actions=frozendict(actions),
    )
    sc.cache[gn] = res
    return res


def solve_equilibria(
    sc: SolvingContext[X, U, Y, RP, RJ],
    gn: GameNode[X, U, Y, RP, RJ],
    solved: Mapping[Mapping[PlayerName, ASet[U]], ASet[Outcome[RP, RJ]]],
):
    if not gn.moves:
        msg = 'Cannot solve_equilibria if there are no moves '
        raise ZValueError(msg, gn=replace(gn, outcomes=None))
    # logger.info(gn=gn, solved=solved)
    # logger.info(possibilities=list(solved))
    players_active = set(gn.moves)
    preferences: Dict[PlayerName, Preference[ASet[Outcome[RP, RJ]]]]
    preferences = {}
    for player_name in players_active:
        preferences[player_name] = PlayerPreferences(sc.gp, player_name, SetPreference1)
    try:
        ea = analyze_equilibria(solved, preferences)
    except ZValueError as e:
        raise ZValueError(moves=gn.moves, states=gn.states) from e
    if len(ea.nondom_nash_equilibria) == 1:
        eq = list(ea.nondom_nash_equilibria)[0]
        game_value = solved[eq]
        actions = eq
        return game_value, actions
    else:
        outcomes = set(ea.nondom_nash_equilibria.values())
        if len(outcomes) == 1:
            game_value = list(outcomes)[0]
            actions = set(ea.nondom_nash_equilibria)
            return game_value, actions
        msg = 'Multiple Nash Equilibria'
        raise ZNotImplementedError(msg, ea=ea)
    raise Exception()


M = TypeVar("M")


class PlayerPreferences(Preference[ASet[Outcome[RP, RJ]]]):
    pt: Preference[ASet[Combined[RJ, RP]]]
    name: PlayerName
    gp: GamePreprocessed

    def __init__(
        self,
        gp: GamePreprocessed,
        name: PlayerName,
        transformer: Callable[[Preference[M]], Preference[ASet[M]]],
    ):
        self.gp = gp
        self.name = name

        p = self.gp.game.players[self.name].preferences
        self.pt = transformer(p)

    def __repr__(self):
        d = {'name': self.name, 'pt': self.pt}
        return 'PlayerPrefs: ' + debug_print(d)
    def transform_to_private(self, a: ASet[Outcome[RP, RJ]]) -> ASet[Combined[RJ, RP]]:
        res = set()
        for s in a:
            check_isinstance(a, Outcome, _self=self)
            p = Combined(joint=s.joint.get(self.name, None), personal=s.private[self.name])
            # logger.info(p=p)
            res.add(p)
        return frozenset(res)

    def compare(self, a: ASet[Outcome[RP, RJ]], b: ASet[Outcome[RP, RJ]]) -> ComparisonOutcome:
        """ <= for the poset """
        check_isinstance(a, set, _self=self)
        check_isinstance(b, set, _self=self)
        a_p = self.transform_to_private(a)
        b_p = self.transform_to_private(b)
        res = self.pt.compare(a_p, b_p)
        assert res in COMP_OUTCOMES, (res, self.pt)
        return res


def solve_final_joint(gn: GameNode[X, U, Y, RP, RJ]) -> Tuple[ASet[Outcome], Mapping]:
    game_value = frozenset({Outcome(private=frozendict(), joint=gn.joint_final_rewards)})
    actions = frozendict()
    return game_value, actions


def solve_final_personal_both(gn: GameNode[X, U, Y, RP, RJ]) -> Tuple[ASet[Outcome], Mapping]:
    game_value = frozenset({Outcome(private=gn.is_final, joint=frozendict())})
    actions = frozendict()
    return game_value, actions


def solve_1_player_final(gn: GameNode[X, U, Y, RP, RJ]) -> Tuple[ASet[Outcome], Mapping]:
    p = list(gn.states)[0]
    # logger.info(f"final for {p}")
    game_value = frozenset({Outcome(private=frozendict({p: gn.is_final[p]}), joint=frozendict())})
    actions = frozendict()
    return game_value, frozendict(actions)


def solve_1_player(
    sc: SolvingContext[X, U, Y, RP, RJ],
    gn: GameNode[X, U, Y, RP, RJ],
    solved: Mapping[Mapping[PlayerName, ASet[U]], ASet[Outcome[RP, RJ]]],
) -> Tuple[ASet[Outcome], Mapping]:
    p = list(gn.states)[0]
    action = list(gn.moves[p])[0]
    actions = frozendict({p: frozenset({action})})
    # incremental = gn.incremental[p][action]
    value = solved[actions]
    game_value = frozenset(value)
    return game_value, actions


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
