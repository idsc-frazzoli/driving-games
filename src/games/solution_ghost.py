import itertools
from dataclasses import dataclass, replace
from typing import Dict, Mapping

from frozendict import frozendict

from possibilities import Poss
from zuper_commons.types import ZValueError
from . import logger
from .comb_utils import valmap
from .game_def import AgentBelief, Game, JointPureActions, JointState, PlayerName, Pr, RJ, RP, SR, U, X, Y
from .structures_solution import GameGraph, GameNode
from .utils import iterate_dict_combinations


def get_ghost_tree(
    game: Game[Pr, X, U, Y, RP, RJ, SR],
    player_name: PlayerName,
    game_graph: GameGraph[Pr, X, U, Y, RP, RJ, SR],
    controllers: Mapping[PlayerName, AgentBelief[Pr, X, U]],
) -> GameGraph[Pr, X, U, Y, RP, RJ, SR]:
    assert len(controllers) >= 1, controllers
    assert player_name not in controllers, (player_name, set(controllers))

    roc = ROContext(game, controllers, dreamer=player_name)
    logger.info(player_name=player_name, controllers=list(controllers))
    state2node = {}
    for k, node in game_graph.state2node.items():
        state2node[k] = replace_others(roc, node)

    return replace(game_graph, state2node=state2node)


@dataclass
class ROContext:
    game: Game[Pr, X, U, Y, RP, RJ, SR]
    # cache: Dict[GameNode[Pr, X, U, Y, RP, RJ, SR], GameNode[Pr, X, U, Y, RP, RJ, SR]]
    controllers: Mapping[PlayerName, AgentBelief[Pr, X, U]]
    dreamer: PlayerName


def replace_others(
    roc: ROContext, node: GameNode[Pr, X, U, Y, RP, RJ, SR],
) -> GameNode[Pr, X, U, Y, RP, RJ, SR]:
    ps = roc.game.ps

    # what would the fixed ones do?
    # evaluate the results
    action_fixed: Dict[PlayerName, Poss[U, Pr]] = {}
    for player_name in node.states:
        # for player_name, controller in controllers.items():
        if player_name in node.is_final:
            continue
        if player_name in node.joint_final_rewards:
            continue
        if player_name == roc.dreamer:
            continue
        state_self = node.states[player_name]
        state_others: JointState = frozendict({k: v for k, v in node.states.items() if k != player_name})
        istate = roc.game.ps.lift_one(state_others)
        options = roc.controllers[player_name].get_commands(state_self, istate)
        # if len(options) != 1:
        #     raise ZNotImplementedError(options=options)
        action_fixed[player_name] = options

    still_moving = set(node.moves) - set(action_fixed)
    # now we redo everything:

    res: Dict[JointPureActions, Poss[Mapping[PlayerName, JointState], Pr]] = {}

    players = list(still_moving)
    CONTEMPLATE = "contemplate"
    new_moves = {}

    for player_name, player_moves in node.moves.items():
        if player_name in still_moving:
            new_moves[player_name] = player_moves
        else:
            new_moves[player_name] = frozenset({CONTEMPLATE})

    new_incremental = {}
    for player_name, player_costs in node.incremental.items():
        if player_name in still_moving:
            new_incremental[player_name] = player_costs
        else:
            identity_cost = roc.game.players[player_name].personal_reward_structure.personal_reward_identity()
            # FIXME: use true cost, but need to have the model include a distribution of costs
            new_incremental[player_name] = frozendict({CONTEMPLATE: identity_cost})

    new_moves = frozendict(new_moves)
    # if players
    if new_moves:
        for active_pure_action in iterate_dict_combinations(new_moves):

            active_mixed: Dict[PlayerName, Poss[U, Pr]]
            active_mixed = valmap(ps.lift_one, active_pure_action)
            active_mixed.update(action_fixed)

            # find out which actions are compatible
            def f(a: JointPureActions) -> Poss[JointState, Pr]:
                if a not in node.outcomes:
                    raise ZValueError(
                        msg, a=a, node=node, active_pure_action=active_pure_action, av=set(node.outcomes)
                    )
                nodes2: Poss[JointState, Pr] = node.outcomes[a]
                return nodes2

            m: Poss[JointState, Pr] = ps.flatten(ps.build_multiple(active_mixed, f))

            res[active_pure_action] = m
    # moves = frozendict(keyfilter(still_moving.__contains__, node.moves))

    ret: GameNode[Pr, X, U, Y, RP, RJ, SR]
    try:
        ret = GameNode(
            states=node.states,
            moves=new_moves,
            outcomes=frozendict(res),
            is_final=node.is_final,
            incremental=frozendict(new_incremental),
            joint_final_rewards=node.joint_final_rewards,
            resources=node.resources,
        )
    except ZValueError as e:
        raise ZValueError("cannot translate", node=node,) from e
    # roc.cache[node] = ret
    return ret
