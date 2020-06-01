from typing import Sequence

import numpy as np
from games import STRATEGY_MIX, STRATEGY_SECURITY, preprocess_game, solve1
from games_scripts import solvers_zoo
from handcrafted_games.handcrafted_game import get_handcrafted_game_spec
import nashpy as nash

"""
Two stages game. After the first stage we could be in 4 possible stages. 
The second stage consists of 4 different games with possible  
with 4 arbitrary payoff matrices for the second stage:
"""


def run_handcrafted_games(max_stages: int, leaves_payoffs: Sequence[np.ndarray]):
    assert max_stages == 2, max_stages
    strategies = [STRATEGY_MIX, STRATEGY_SECURITY]
    for strategy in strategies:
        solver_spec = solvers_zoo["solver-1-" + strategy]
        solver_params = solver_spec.solver_params
        game_spec = get_handcrafted_game_spec(max_stages, leaves_payoffs)
        game_preprocessed = preprocess_game(game_spec.game, solver_params)
        solutions = solve1(game_preprocessed)
        # todo check solutions with what we expect
        print(solutions)


if __name__ == "__main__":
    max_stages = 2
    G1 = np.array([[[1, 0], [1, 2]], [[3, 2], [5, 5]]])
    G2 = np.array([[[1, 0], [4, 1]], [[2, 3], [2, 1]]])
    G3 = np.array([[[1, 4], [3, 1]], [[2, 1], [2, 2]]])
    G4 = np.array([[[3, 3], [4, 3]], [[3, 2], [5, 3]]])
    leaves_payoffs = (G1, G2, G3, G4)
    for i, G in enumerate(leaves_payoffs):
        print(
            "Game G{} equilibria: ".format(i + 1),
            list(nash.Game(-G[:, :, 0], -G[:, :, 1]).support_enumeration()),
        )
    run_handcrafted_games(max_stages, leaves_payoffs)
