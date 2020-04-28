from dataclasses import dataclass
from decimal import Decimal as D
from typing import Dict, Generic

from frozendict import frozendict

from .game_def import (
    GamePlayer,
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
