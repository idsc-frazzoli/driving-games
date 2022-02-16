from datetime import timedelta
from itertools import product
from time import perf_counter

from parameterized import parameterized

from driving_games import VehicleTrackState
from driving_games.collisions_check import joint_collision_cost_simple
from driving_games.demo import without_compmake
from driving_games.resources import get_poly_occupancy
from driving_games.zoo_games import games_zoo
from driving_games.zoo_solvers import solvers_zoo
from driving_games_tests import logger

do_games_sets = [
    "4way_int_2p_sets",
    "4way_int_3p_sets",
    # "4way_int_3p_sets",
    # "complex_int_6p_sets"
]
do_solvers_sets = [
    "solver-2-pure-security_mNE-naive-noextra",
    "solver-2-pure-security_mNE-fact-noextra"
    # "solver-1-pure-security_mNE-fact",
    # "solver-1-finite_mix-mix_mNE-fact",
    # "solver-1-finite_mix-security_mNE-naive",
    # "solver-1-finite_mix-security_mNE-fact",
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


do_games_prob = ["4way_int_2p_prob", "4way_int_3p_prob"]
do_solvers_prob = [
    "solver-1-pure-mix_mNE-fact-extra",
    # "solver-1-security-naive",
    # "solver-1-finite_mix-mix_mNE-fact",
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
