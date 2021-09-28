import unittest
from itertools import product
from parameterized import parameterized
from driving_games import uncertainty_prob, uncertainty_sets
from games import MIX_MNE, SECURITY_MNE, UncertaintyParams, FINITE_MIX_STRATEGIES, PURE_STRATEGIES
from games_zoo import solvers_zoo
from games_zoo.solvers import SolverSpec
from _tmp.toy_games import ToyGameMat
import nashpy as nash
from _tmp.toy_games_tests.run_toy_games import _run_toy_game, _run_toy_game_bayesian
from _tmp.toy_games_tests.toy_games_tests_zoo import *

games = (game2, game21, game3, game4, game5)  # todo fix game1 is problematic for now
admissible_strategies = [PURE_STRATEGIES, FINITE_MIX_STRATEGIES]
strategies_mne = [MIX_MNE, SECURITY_MNE]

solvers = (
    solvers_zoo["solver-1-" + strat + "-" + mne_strat + "-naive"]
    for strat, mne_strat in product(*[admissible_strategies, strategies_mne])
)
uncertainties = [uncertainty_sets, uncertainty_prob]
toy_tests = list(product(games, solvers, uncertainties))


@parameterized(toy_tests)
def test_toy_games(toygame: ToyGameMat, solver_spec: SolverSpec, uncertainty_params: UncertaintyParams):
    """Test Toy Game"""
    logger.info(f"Toygame description: {toygame.desc}")
    for i, G in enumerate(toygame.subgames):
        subgames_solutions = list(nash.Game(-G.A, -G.B).vertex_enumeration())
        logger.info(f"Game G{i + 1} equilibria: {subgames_solutions}")
    _run_toy_game(toygame, solver_spec, uncertainty_params)
    logger.info("Completed toy game test")


def test_prob_debug():
    game = game2
    solver_spec = solvers_zoo["solver-1-finite_mix-mix_mNE-naive"]
    uncertainty_params = uncertainty_prob
    _run_toy_game(game, solver_spec, uncertainty_params)


@unittest.skip("Bayesian games to be refactored in the future")
def test_bayesian_debug():
    game = game7
    solver_spec = solvers_zoo["solver-1-finite_mix-mix_mNE-naive"]
    uncertainty_params = uncertainty_prob
    _run_toy_game_bayesian(game.subgames, solver_spec, uncertainty_params)
