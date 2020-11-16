from decimal import Decimal as D
from typing import AbstractSet, Mapping, Optional, MutableMapping
from bayesian_driving_games.structures import BayesianGame, PlayerType


from bayesian_driving_games.structures_solution import BayesianGameNode, BayesianGameGraph
from games.create_joint_game_tree import create_game_graph
from possibilities import Poss
from games.game_def import (
    JointState,
    PlayerName,
    X,
)
from games.structures_solution import (
    GameFactorization,
    GameGraph,
)

__all__ = ["create_bayesian_game_graph"]


def _initialize_bayesian_game_graph(game: BayesianGame, game_graph: GameGraph) -> BayesianGameGraph:
    """Substitute standard `GameNode` with `BayesianGameNode`.
    Copy plus adding the game belief for players at that node.
    Very inefficient at the moment.
    """
    state2bnode: MutableMapping[JointState, BayesianGameNode] = dict()
    for js, gnode in game_graph.state2node.items():
        game_node_belief: MutableMapping[PlayerName, Mapping[PlayerName, Poss[PlayerType]]] = dict()
        for player in js:
            game_node_belief[player] = game.players[player].prior

        state2bnode[js] = BayesianGameNode(
            states=gnode.states,
            moves=gnode.moves,
            outcomes=gnode.outcomes,
            is_final=gnode.is_final,
            incremental=gnode.incremental,
            joint_final_rewards=gnode.joint_final_rewards,
            resources=gnode.resources,
            # fixme the game belif needs to be frozen?
            game_node_belief=game_node_belief,
        )

    return BayesianGameGraph(initials=game_graph.initials, state2node=state2bnode, ti=game_graph.ti)


def create_bayesian_game_graph(
    game: BayesianGame,
    dt: D,
    initials: AbstractSet[JointState],
    gf: Optional[GameFactorization[X]],
) -> BayesianGameGraph:
    """Create the game graph.

    :param game: Game parameters
    :param dt: timestep
    :param initials: Initial states of the players
    :param gf: Game factorization (optional)
    :return: Returns the game graph of the bayesian game, consisting of BayesianGameNodes including Beliefs.
    """
    game_graph = create_game_graph(game=game, dt=dt, initials=initials, gf=gf)
    bayesian_game_graph = _initialize_bayesian_game_graph(game, game_graph)
    return bayesian_game_graph
