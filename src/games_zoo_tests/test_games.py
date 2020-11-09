from datetime import timedelta
from itertools import product
from time import perf_counter

from parameterized import parameterized

from games_zoo import games_zoo, solvers_zoo
from games_zoo.demo import without_compmake

from games_zoo_tests import logger

do_games_sets = [
    "sym_v1_sets",
    "asym_v1_sets",
    "asym_v0_sets",
]
do_solvers_sets = [
    "solver-1-mix-naive",
    "solver-1-security-naive",
    "solver-1-security-fact",
]
games_sets = {k: games_zoo[k] for k in do_games_sets}
solvers_sets = {k: solvers_zoo[k] for k in do_solvers_sets}
games_n_solvers_sets = list(product(games_sets, solvers_sets))


@parameterized(games_n_solvers_sets)
def test_games_sets(game_str: str, solver_str: str) -> None:
    logger.info(f"Starting game test:\n\tName: {game_str}\n\tSolver:{solver_str}")
    game = {game_str: games_zoo[game_str]}
    solver = {solver_str: solvers_zoo[solver_str]}
    res = without_compmake(game, solver)


do_games_prob = [  # "sym_v1_prob",
    # "asym_v0_prob",
    "asym_v1_prob"
]
do_solvers_prob = [
    "solver-1-mix-fact",
    # "solver-1-security-naive",
    "solver-1-security-fact",
]
games_prob = {k: games_zoo[k] for k in do_games_prob}
solvers_prob = {k: solvers_zoo[k] for k in do_solvers_prob}
games_n_solvers_prob = list(product(games_prob, solvers_prob))


@parameterized(games_n_solvers_prob)
def test_games_prob(game_str: str, solver_str: str) -> None:
    logger.info(f"Starting game test:\n\tName: {game_str}\n\tSolver: {solver_str}")
    games = {game_str: games_zoo[game_str]}
    solvers = {solver_str: solvers_zoo[solver_str]}
    tic = perf_counter()
    res = without_compmake(games, solvers)
    toc = timedelta(seconds=perf_counter() - tic)
    logger.info(f"It took: {toc}")