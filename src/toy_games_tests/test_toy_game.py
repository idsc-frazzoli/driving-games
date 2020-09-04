from itertools import product
from typing import Sequence

import numpy as np
from parameterized import parameterized

from driving_games import uncertainty_prob, uncertainty_sets, TwoVehicleUncertaintyParams
from games import STRATEGY_MIX, STRATEGY_SECURITY, preprocess_game, solve1, PlayerName
from games_scripts import solvers_zoo
from games_scripts.solvers import SolverSpec
from nash.structures import print_bimatgame, BiMatGame
from toy_games.toy_rewards import BirdJointReward
from toy_games_tests import logger
from toy_games.toy_game import get_toy_game_spec
import nashpy as nash

from toy_games_tests.toy_games_tests_zoo import game1, game2, ToyGame, game3, game4

"""
Two stages game. After the first stage we could be in 4 possible stages. 
The second stage consists of 4 different games with possible  
with 4 arbitrary payoff matrices for the second stage:
"""


def _run_toy_game(
    subgames: Sequence[BiMatGame], solver_spec: SolverSpec, uncertainty_params: TwoVehicleUncertaintyParams,
):
    max_stages = 1
    p1_name, p2_name = PlayerName("1"), PlayerName("2")

    logger.info("Starting a 2 stage toy game with the following subgames:")
    for i, bimatgame in enumerate(subgames):
        logger.info("Subgame {}: {}".format(i, print_bimatgame(bimatgame)))

    solver_params = solver_spec.solver_params
    game_spec = get_toy_game_spec(max_stages, subgames, uncertainty_params)
    game = game_spec.game
    game_preprocessed = preprocess_game(game, solver_params)
    solutions = solve1(game_preprocessed)
    for state, solution in solutions.game_solution.states_to_solution.items():
        # filter out only the first level subgame
        if all([p.stage == 1 for p in state.values()]):
            game_idx, _, _ = BirdJointReward.get_payoff_matrix_idx(state[p1_name], state[p2_name])
            # print("Game solution of game:", gamemat2str(leaves_payoffs[game_idx]))
            logger.info("Joint state:\n", state)
            logger.info("Values and actions:\n", solution.solved)
            logger.info("Game values:\n", solution.va.game_value)

    # todo check solutions with what we expect
    # todo create report/visualisation
    logger.info(solutions)


games = (game1, game2)
strategies = [STRATEGY_MIX, STRATEGY_SECURITY]
solvers = (solvers_zoo["solver-1-" + strategy + "-naive"] for strategy in strategies)
uncertainties = [uncertainty_sets, uncertainty_prob]
toy_tests = list(product(games, solvers, uncertainties))


@parameterized(toy_tests)
def test_toy_games(
    toygame: ToyGame, solver_spec: SolverSpec, uncertainty_params: TwoVehicleUncertaintyParams
):
    for i, G in enumerate(toygame.subgames):
        logger.info(
            "Game G{} equilibria: ".format(i + 1), list(nash.Game(-G.A, -G.B).vertex_enumeration()),
        )
    _run_toy_game(toygame.subgames, solver_spec, uncertainty_params)
    logger.info("Completed toy game test")


def test_prob_debug():
    game = game4
    solver_spec = solvers_zoo["solver-1-mix-naive"]
    uncertainty_params = uncertainty_prob
    _run_toy_game(game.subgames, solver_spec, uncertainty_params)
