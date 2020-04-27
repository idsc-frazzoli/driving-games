from dataclasses import dataclass
from decimal import Decimal as D
from typing import Dict, Generic

from frozendict import frozendict
from networkx import MultiDiGraph
from zuper_commons.types import ZException

from . import logger
from .game_def import (
    ASet,
    Dynamics,
    GamePlayer,
    PersonalRewardStructure,
    PlayerName,
    RJ,
    RP,
    U,
    X,
    Y,
)
from .structures_solution import GameNode


@dataclass
class P1Context(Generic[X, U, Y, RP, RJ]):
    cache: Dict[X, GameNode[X, U, Y, RP, RJ]]
    dt: D


def get_one_player_game_tree(
    *, player_name: PlayerName, player: GamePlayer[X, U, Y, RP, RJ], x0: X, dt: D
) -> GameNode[X, U, Y, RP, RJ]:
    context = P1Context({}, dt)
    return get_1p_game_tree(c=context, player_name=player_name, player=player, x0=x0)


def get_1p_game_tree(
    *,
    c: P1Context[X, U, Y, RP, RJ],
    player_name: PlayerName,
    player: GamePlayer[X, U, Y, RP, RJ],
    x0: X,
) -> GameNode[X, U, Y, RP, RJ]:
    assert not isinstance(x0, set), x0
    prs = player.personal_reward_structure
    dyn = player.dynamics

    states = frozendict({player_name: x0})

    is_final_state = prs.is_personal_final_state(x0)
    if is_final_state:
        moves = frozendict()
        outcomes = frozendict()
        incremental = frozendict()
        is_final = frozendict({player_name: prs.personal_final_reward(x0)})

    else:
        is_final = frozendict()
        successors = dyn.successors(x0, c.dt)
        moves = frozendict({player_name: frozenset(successors)})
        inc = frozendict({u: prs.personal_reward_incremental(x0, u, c.dt) for u in successors})
        incremental = frozendict({player_name: inc})
        outcomes = {}
        for u, x1s in successors.items():
            actions = frozendict({player_name: u})
            x1 = list(x1s)[0]  # XXX: no multimodal
            outcomes[actions] = get_1p_game_tree(c=c, player_name=player_name, player=player, x0=x1)

        outcomes = frozendict(outcomes)
    joint_final_rewards = frozendict()

    res = GameNode(
        states=states,
        moves=moves,
        outcomes=outcomes,
        is_final=is_final,
        incremental=incremental,
        joint_final_rewards=joint_final_rewards,
    )
    c.cache[x0] = res
    return res


#    states: Mapping[PlayerName, X]
#     moves: Mapping[PlayerName, ASet[U]]
#     outcomes: "Mapping[Mapping[PlayerName, ASet[U]], GameNode[X, U, Y, RP, RJ]]"
#
#     is_final: Mapping[PlayerName, RP]
#     incremental: Mapping[PlayerName, Mapping[U, RP]]
#
#     joint_final_rewards: Mapping[PlayerName, RJ]


def get_accessible_states(
    initial: ASet[X],
    personal_reward_structure: PersonalRewardStructure[X, U, RP],
    dynamics: Dynamics[X, U],
    dt: D,
) -> MultiDiGraph:
    G = MultiDiGraph()

    for i in initial:
        i_final = personal_reward_structure.is_personal_final_state(i)
        if i_final:
            raise ZException(i_final=i_final)

        G.add_node(i, is_final=False)
    stack = list(initial)
    logger.info(stack=stack)
    i = 0
    expanded = set()
    while stack:
        # print(i, len(stack), len(G.nodes))
        i += 1
        s1 = stack.pop(0)
        assert s1 in G.nodes
        if s1 in expanded:
            continue
        # is_final =  player.personal_reward_structure.is_personal_final_state(s1)
        # G.add_node(s1, is_final=is_final)
        # # logger.info(s1=G.nodes[s1])

        expanded.add(s1)
        successors = dynamics.successors(s1, dt)
        for u, s2s in successors.items():
            for s2 in s2s:
                if s2 not in G.nodes:
                    is_final2 = personal_reward_structure.is_personal_final_state(s2)
                    G.add_node(s2, is_final=is_final2)
                    if not is_final2:
                        stack.append(s2)

                G.add_edge(s1, s2, u=u)
    return G
