from itertools import product

from parameterized import parameterized

from games_zoo import games_zoo, solvers_zoo
from games_zoo.demo import without_compmake

from games_zoo_tests import logger


do_games = [
    "sym_v1_sets",
    "asym_v1_sets",
    "asym_v0_sets",
]  # "sym_v1_prob","asym_v0_prob","sym_v1_prob" # fixme include prob tests once they are working
do_solvers = ["solver-1-mix-naive", "solver-1-security-naive", "solver-1-security-fact"]
games = {k: games_zoo[k] for k in do_games}
solvers = {k: solvers_zoo[k] for k in do_solvers}
games_n_solvers = list(product(games, solvers))


@parameterized(games_n_solvers)
def test_game(game_str: str, solver_str: str) -> None:
    logger.info(f"Starting game test:\n\tName: {game_str}\n\tSolver:{solver_str}")
    game = {game_str: games_zoo[game_str]}
    solver = {solver_str: solvers_zoo[solver_str]}
    res = without_compmake(game, solver)


def test_run4_prob_tmp() -> None:
    do_games = ["asym_v0_prob"]
    do_solvers = [
        "solver-1-security-fact",
        "solver-1-mix-naive"
    ]
    games = {k: games_zoo[k] for k in do_games}
    solvers = {k: solvers_zoo[k] for k in do_solvers}
    res = without_compmake(games, solvers)
