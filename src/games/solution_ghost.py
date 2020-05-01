import itertools
from dataclasses import dataclass
from typing import Dict, Mapping

from frozendict import frozendict
from toolz import keyfilter

from games import GameNode, GamePreprocessed, JointPureActions, JointState, PlayerName, RJ, RP, U, X, Y
from games.comb_utils import valmap
from games.game_def import AgentBelief, Pr
from possibilities import Poss
from zuper_commons.types import ZValueError


def get_ghost_tree(
    gp: GamePreprocessed,
    player_name: PlayerName,
    game_tree: GameNode[Pr, X, U, Y, RP, RJ],
    controllers: Mapping[PlayerName, AgentBelief[Pr, X, U]],
) -> GameNode[Pr, X, U, Y, RP, RJ]:
    assert len(controllers) >= 1, controllers
    assert player_name not in controllers, (player_name, set(controllers))

    roc = ROContext(gp, {}, controllers, dreamer=player_name)
    return replace_others(roc, game_tree)


@dataclass
class ROContext:
    gp: GamePreprocessed
    cache: Dict[GameNode[Pr, X, U, Y, RP, RJ], GameNode[Pr, X, U, Y, RP, RJ]]
    controllers: Mapping[PlayerName, AgentBelief[Pr, X, U]]
    dreamer: PlayerName


def replace_others(roc: ROContext, node: GameNode[Pr, X, U, Y, RP, RJ], ) -> GameNode[Pr, X, U, Y, RP, RJ]:
    ps = roc.gp.game.ps
    if node in roc.cache:
        return roc.cache[node]
    assert roc.dreamer not in roc.controllers
    assert roc.controllers
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
        istate = roc.gp.game.ps.lift_one(state_others)
        options = roc.controllers[player_name].get_commands(state_self, istate)
        # if len(options) != 1:
        #     raise ZNotImplementedError(options=options)
        action_fixed[player_name] = options

    still_moving = set(node.moves) - set(action_fixed)
    # now we redo everything:

    res: Dict[JointPureActions, Poss[GameNode[Pr, X, U, Y, RP, RJ], Pr]] = {}

    players = list(still_moving)

    if players:
        choices = [node.moves[_] for _ in players]
        for _ in itertools.product(*tuple(choices)):
            active_pure_action = frozendict(zip(players, _))
            active_mixed: Dict[PlayerName, Poss[U, Pr]]
            active_mixed = valmap(ps.lift_one, active_pure_action)
            active_mixed.update(action_fixed)

            # find out which actions are compatible
            def f(a: JointPureActions) -> Poss[GameNode[Pr, X, U, Y, RP, RJ], Pr]:
                if a not in node.outcomes2:
                    raise ZValueError(a=a, node=node, choices=choices, av=set(node.outcomes2))
                nodes2: Poss[GameNode, Pr] = node.outcomes2[a]

                def g(r: GameNode) -> GameNode:
                    return replace_others(roc, r)

                return ps.build(nodes2, g)

            m: Poss[GameNode[Pr, X, U, Y, RP, RJ], Pr] = ps.flatten(ps.build_multiple(active_mixed, f))
            res[active_pure_action] = m
    moves = frozendict(keyfilter(still_moving.__contains__, node.moves))

    ret: GameNode[Pr, X, U, Y, RP, RJ]
    ret = GameNode(
        states=node.states,
        moves=moves,
        outcomes2=frozendict(res),
        is_final=node.is_final,
        incremental=node.incremental,
        joint_final_rewards=node.joint_final_rewards,
    )
    roc.cache[node] = ret
    return ret
    #
    #
    # compatible: Mapping[JointPureActions, Poss[GameNode[Pr, X, U, Y, RP, RJ], Pr]]
    # compatible = {k: v for k, v in node.outcomes2.items()
    #               if is_compatible(k, action_others)}
    #
    #
    #
    # outcomes = Mapping[JointPureActions, Poss[GameNode[Pr, X, U, Y, RP, RJ], Pr]]
    # outcomes = {
    #     k: replace_others(dreamer, v, controllers, cache)
    #     for k, v in node.outcomes.items()
    #     if is_compatible(k, action_others)
    # }

    # logger.info(action_others=action_others, original=set(node.outcomes), compatible=set(outcomes))

    # moves = get_all_choices_by_players(set(outcomes))
    # for player_name in action_others:
    #     if len(moves[player_name]) != 1:
    #         raise ZValueError(
    #             moves=moves, dreamer=dreamer, controllers=list(controllers), orig_moves=node.moves
    #         )

    # if len(moves) == len(node.moves):
    #     raise ZValueError(moves=moves, dreamer=dreamer, controllers=list(controllers), orig_moves=node.moves)
