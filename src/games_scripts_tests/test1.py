from games_scripts import games_zoo, solvers_zoo
from games_scripts.demo import without_compmake


def test_run1() -> None:
    do_games = ["sym_v1"]
    do_solvers = [
        "solver-1-mix",
    ]

    games = {k: games_zoo[k] for k in do_games}
    solvers = {k: solvers_zoo[k] for k in do_solvers}
    res = without_compmake(games, solvers)


def test_run2() -> None:
    do_games = ["sym_v1"]
    do_solvers = [
        "solver-1-security",
    ]

    games = {k: games_zoo[k] for k in do_games}
    solvers = {k: solvers_zoo[k] for k in do_solvers}
    res = without_compmake(games, solvers)


def run3() -> None:
    do_games = ["asym_v0"]
    do_solvers = [
        "solver-1-security",
    ]

    games = {k: games_zoo[k] for k in do_games}
    solvers = {k: solvers_zoo[k] for k in do_solvers}
    res = without_compmake(games, solvers)


if __name__ == "__main__":
    # test_run2()
    run3()
