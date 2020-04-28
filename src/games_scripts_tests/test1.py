from games_scripts import games_zoo, solvers_zoo

#
# def test_run1():
#     args = ["-o", "out/tests/test_run1"]
#     ret = dg_demo(args=args, sys_exit=False)
#     if ret:  # pragma: no cover
#         raise ZValueError(ret=ret)
from games_scripts.demo import without_compmake


def test_run1():
    do_games = ["game1"]
    do_solvers = ["solver1"]
    games = {k: games_zoo[k] for k in do_games}
    solvers = {k: solvers_zoo[k] for k in do_solvers}
    res = without_compmake(games, solvers)
