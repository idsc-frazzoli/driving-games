from dataclasses import dataclass
from decimal import Decimal as D
from typing import Dict, Generic, Set

from frozendict import frozendict

from zuper_commons.types import ZValueError
from . import logger
from .game_def import Game, GamePlayer, PlayerName, Pr, RJ, RP, U, X, Y
from .structures_solution import GameNode

__all__ = []


@dataclass
class P1Context(Generic[Pr, X, U, Y, RP, RJ]):
    cache: Dict[X, GameNode[Pr, X, U, Y, RP, RJ]]
    dt: D
    processing: Set[X]


def get_one_player_game_tree(
    *, game: Game, player_name: PlayerName, player: GamePlayer[Pr, X, U, Y, RP, RJ], x0: X, dt: D
) -> GameNode[Pr, X, U, Y, RP, RJ]:
    context = P1Context({}, dt, set())
    return get_1p_game_tree(game=game, c=context, player_name=player_name, player=player, x0=x0)


def get_1p_game_tree(
    *,
    game: Game,
    c: P1Context[Pr, X, U, Y, RP, RJ],
    player_name: PlayerName,
    player: GamePlayer[Pr, X, U, Y, RP, RJ],
    x0: X,
) -> GameNode[Pr, X, U, Y, RP, RJ]:
    if x0 in c.cache:
        return c.cache[x0]
    if x0 in c.processing:
        msg = 'Found loop'
        raise ZValueError(msg, x0=x0, processing=c.processing)
    c.processing.add(x0)
    # logger.info(x0=x0)
    # logger.info('game tree', x0=x0)
    assert not isinstance(x0, set), x0
    prs = player.personal_reward_structure
    dyn = player.dynamics
    ps = game.ps

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
            # logger.info(successors=successors)
            r = ps.build(
                x1s, lambda _: get_1p_game_tree(game=game, c=c, player_name=player_name, player=player, x0=_)
            )

            outcomes[actions] = r

        outcomes = frozendict(outcomes)
    joint_final_rewards = frozendict()

    res = GameNode(
        states=states,
        moves=moves,
        outcomes2=outcomes,
        is_final=is_final,
        incremental=incremental,
        joint_final_rewards=joint_final_rewards,
    )
    c.cache[x0] = res
    c.processing.remove(x0)
    return res
