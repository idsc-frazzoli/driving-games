from itertools import product

from parametrized import parametrized

from games_scripts import games_zoo, solvers_zoo
from games_scripts.demo import without_compmake

from games_scripts_tests import logger

# fixme include prob tests once they are working
do_games = [
    "sym_v1_sets",
    "sym_v1_sets",
    "asym_v0_sets",
]  # "sym_v1_prob","asym_v0_prob","sym_v1_prob"
do_solvers = ["solver-1-mix-naive", "solver-1-security-naive", "solver-1-security-fact"]
games = {k: games_zoo[k] for k in do_games}
solvers = {k: solvers_zoo[k] for k in do_solvers}
games_n_solvers = product(games, solvers)


@parametrized(games_n_solvers)
def test_game(game_str: str, solver_str: str) -> None:
    logger.info("Starting game test:\n\tName: {}\n\tSolver:{}".format(game_str, solver_str))
    game = {game_str: games_zoo[game_str]}
    solver = {solver_str: solvers_zoo[solver_str]}
    res = without_compmake(game, solver)


def test_run4_prob_tmp() -> None:
    do_games = ["sym_v1_prob"]
    do_solvers = [
        # "solver-1-security-fact",
        "solver-1-security-naive"
    ]
    games = {k: games_zoo[k] for k in do_games}
    solvers = {k: solvers_zoo[k] for k in do_solvers}
    res = without_compmake(games, solvers)
