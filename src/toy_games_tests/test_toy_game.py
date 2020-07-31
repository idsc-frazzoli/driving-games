from itertools import product
from typing import Sequence, List

import numpy as np
from parameterized import parameterized, param

from games import STRATEGY_MIX, STRATEGY_SECURITY, preprocess_game, solve1, PlayerName
from games_scripts import solvers_zoo
from games_scripts.solvers import SolverSpec
from toy_games_tests import logger
from toy_games.toy_game import get_toy_game_spec
import nashpy as nash

from toy_games_tests.toy_games_tests_zoo import game1, game2, ToyGame

"""
Two stages game. After the first stage we could be in 4 possible stages. 
The second stage consists of 4 different games with possible  
with 4 arbitrary payoff matrices for the second stage:
"""


def _gamemat2str(mat: np.ndarray) -> str:
    assert mat.shape[0] == 2
    rows, cols = mat.shape[1:3]
    str_game = ""
    for r in range(rows):
        str_game += "\n\t"
        for c in range(cols):
            str_game += "{},{}\t\t".format(mat[0, r, c], mat[1, r, c])
    return str_game


def _run_toy_game(leaves_payoffs: Sequence[np.ndarray], solver_spec: SolverSpec):
    max_stages = 2
    p1_name, p2_name = PlayerName("1"), PlayerName("2")

    logger.info("Starting a 2 stage toy game with the following subgames:")
    for i, subgame in enumerate(leaves_payoffs):
        logger.info("Subgame {}: {}".format(i, _gamemat2str(subgame)))

    solver_params = solver_spec.solver_params
    game_spec = get_toy_game_spec(max_stages, leaves_payoffs)
    game = game_spec.game
    game_preprocessed = preprocess_game(game, solver_params)
    solutions = solve1(game_preprocessed)
    for state, solution in solutions.game_solution.states_to_solution.items():
        # filter out only the first level subgame
        if all([p.stage == 1 for p in state.values()]):
            game_idx, _, _ = game.joint_reward.get_payoff_matrix_idx(state[p1_name], state[p2_name])
            print("Game solution of game:", _gamemat2str(leaves_payoffs[game_idx]))
            print("Joint state:\n", state)
            print("Values and actions:\n", solution.solved)
            print("Game values:\n", solution.va.game_value)

    # todo check solutions with what we expect
    # todo create report/visualisation
    logger.info(solutions)


games = (game1, game2)
strategies = [STRATEGY_MIX, STRATEGY_SECURITY]
solvers = (solvers_zoo["solver-1-" + strategy + "-naive"] for strategy in strategies)
toy_tests = list(product(games, solvers))


@parameterized(toy_tests)
def test_toy_games(toygame: ToyGame, solver_spec: SolverSpec):
    # np.set_printoptions(formatter={'float': "\t{: 0.0f}\t".format})
    max_stages = 2

    for i, G in enumerate(toygame.subgames):
        logger.info(
            "Game G{} equilibria: ".format(i + 1),
            list(nash.Game(-G[:, :, 0], -G[:, :, 1]).vertex_enumeration()),
        )
    _run_toy_game(toygame.subgames, solver_spec)
    logger.info("Compleated toy game test")


# if __name__ == "__main__":
#     test_toy_games(game1, solvers.__next__())
