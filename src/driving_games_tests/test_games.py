from datetime import timedelta
from itertools import product
from time import perf_counter

import pytest

from driving_games.demo import without_compmake
from driving_games.zoo_games import games_zoo
from driving_games.zoo_solvers import solvers_zoo
from driving_games_tests import logger

do_games_sets = [
    # "simple_int_2p_sets",
    # "simple_int_3p_sets",
    # "multilane_int_2p_sets",
    # "multilane_int_3p_sets",
    # "multilane_int_4p_sets",
    "multilane_int_5p_sets",
    # "multilane_int_5p_sets",
    # "multilane_int_6p_sets",
    # "complex_int_2p_sets",
    # "complex_int_3p_sets",
    # "complex_int_4p_sets",
    # "complex_int_5p_sets",
    # "complex_int_6p_sets",
]
do_solvers_sets = [
    # "solver-2-pure-security_mNE-naive-extra-inf",
    # "solver-2-pure-security_mNE-fact1-noextra-inf",
    "solver-2-pure-security_mNE-fact2-noextra-inf"
    # "solver-2-pure-security_mNE-fact2-extra"
    # "solver-1-pure-security_mNE-fact",
    # "solver-1-finite_mix-mix_mNE-fact",
    # "solver-1-finite_mix-security_mNE-naive",
    # "solver-1-finite_mix-security_mNE-fact",
]
games_sets = {k: games_zoo[k] for k in do_games_sets}
solvers_sets = {k: solvers_zoo[k] for k in do_solvers_sets}
games_n_solvers_sets = list(product(games_sets, solvers_sets))


@pytest.mark.parametrize("game_str, solver_str", games_n_solvers_sets)
def test_games_sets(game_str: str, solver_str: str) -> None:
    logger.info(f"Starting game test:\n\tName: {game_str}\n\tSolver:{solver_str}")
    game = {game_str: games_zoo[game_str]}
    solver = {solver_str: solvers_zoo[solver_str]}
    res = without_compmake(game, solver)
    # print(joint_simple_collision_cost.cache_info())
    # print(VehicleTrackState.to_global_pose.cache_info())
    # print(get_poly_occupancy.cache_info())


do_games_prob = ["simple_int_2p_prob", "simple_int_3p_prob"]
do_solvers_prob = [
    "solver-1-pure-mix_mNE-fact1-extra-inf",
    # "solver-1-security-naive",
    # "solver-1-finite_mix-mix_mNE-fact",
]
games_prob = {k: games_zoo[k] for k in do_games_prob}
solvers_prob = {k: solvers_zoo[k] for k in do_solvers_prob}
games_n_solvers_prob = list(product(games_prob, solvers_prob))


@pytest.mark.parametrize("game_str, solver_str", games_n_solvers_prob)
def test_games_prob(game_str: str, solver_str: str):
    logger.info(f"Starting game test:\n\tName: {game_str}\n\tSolver: {solver_str}")
    games = {game_str: games_zoo[game_str]}
    solvers = {solver_str: solvers_zoo[solver_str]}
    tic = perf_counter()
    res = without_compmake(games, solvers)
    toc = timedelta(seconds=perf_counter() - tic)
    logger.info(f"It took: {toc}")
