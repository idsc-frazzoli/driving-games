from games_scripts import games_zoo, solvers_zoo

#
# def test_run1():
#     args = ["-o", "out/tests/test_run1"]
#     ret = dg_demo(args=args, sys_exit=False)
#     if ret:  # pragma: no cover
#         raise ZValueError(ret=ret)
from games_scripts.demo import without_compmake


def test_run1() -> None:
    do_games = ["game1"]
    do_solvers = [
        "solver-1-strategy-mix",
    ]

    games = {k: games_zoo[k] for k in do_games}
    solvers = {k: solvers_zoo[k] for k in do_solvers}
    res = without_compmake(games, solvers)


def test_run2() -> None:
    do_games = ["game1"]
    do_solvers = [
        "solver-1-strategy-security",
    ]

    games = {k: games_zoo[k] for k in do_games}
    solvers = {k: solvers_zoo[k] for k in do_solvers}
    res = without_compmake(games, solvers)


if __name__ == "__main__":
    test_run2()
