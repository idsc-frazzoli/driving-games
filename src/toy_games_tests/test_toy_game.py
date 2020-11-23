import unittest
from itertools import product
from parameterized import parameterized
from driving_games import uncertainty_prob, uncertainty_sets, TwoVehicleUncertaintyParams
from games import STRATEGY_MIX, STRATEGY_SECURITY
from games_zoo import solvers_zoo
from games_zoo.solvers import SolverSpec
from toy_games import ToyGameMat
from toy_games_tests import logger
import nashpy as nash
from toy_games_tests.run_toy_games import _run_toy_game, _run_toy_game_bayesian
from toy_games_tests.toy_games_tests_zoo import *

games = (game2, game21, game3, game4, game5)  # game1, # todo fix game1 is problematic for now
strategies = [STRATEGY_MIX, STRATEGY_SECURITY]
solvers = (solvers_zoo["solver-1-" + strategy + "-naive"] for strategy in strategies)
uncertainties = [uncertainty_sets, uncertainty_prob]
toy_tests = list(product(games, solvers, uncertainties))


@parameterized(toy_tests)
def test_toy_games(
    toygame: ToyGameMat, solver_spec: SolverSpec, uncertainty_params: TwoVehicleUncertaintyParams
):
    """Test Toy Game"""
    logger.info(f"Toygame description: {toygame.desc}")
    for i, G in enumerate(toygame.subgames):
        subgames_solutions = list(nash.Game(-G.A, -G.B).vertex_enumeration())
        logger.info(f"Game G{i + 1} equilibria: {subgames_solutions}")
    _run_toy_game(toygame, solver_spec, uncertainty_params)
    logger.info("Completed toy game test")


def test_prob_debug():
    game = game3
    solver_spec = solvers_zoo["solver-1-mix-naive"]
    uncertainty_params = uncertainty_prob
    _run_toy_game(game, solver_spec, uncertainty_params)


@unittest.skip("Bayesian games to be refactored in the future")
def test_bayesian_debug():
    game = game7
    solver_spec = solvers_zoo["solver-1-mix-naive"]
    uncertainty_params = uncertainty_prob
    _run_toy_game_bayesian(game.subgames, solver_spec, uncertainty_params)
