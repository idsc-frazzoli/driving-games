from os.path import join
from typing import Mapping

from decent_params import DecentParams
from quickapp import QuickApp, QuickAppContext
from zuper_commons.text import expand_string
from zuper_commons.types import ZValueError

from .zoo_games import games_zoo
from .zoo_solvers import solvers_zoo, SolverSpec
from games import GameSpec
from games import (
    create_report_preprocessed,
    report_game_visualization,
    report_solutions,
    solve_main,
)
from games.performance import PerformanceStatistics, report_performance_stats
from games.preprocess import preprocess_game

__all__ = ["dg_demo", "DGDemo", "without_compmake"]


class DGDemo(QuickApp):
    """Main function"""

    def define_options(self, params: DecentParams):
        params.add_string("games", default="4way_int_3p_sets")
        params.add_string("solvers", default="solver-2-pure-security_mNE-naive")

    def define_jobs_context(self, context: QuickAppContext):

        do_solvers = self.get_options().solvers.split(",")
        do_games = self.get_options().games.split(",")

        do_games = expand_string(do_games, list(games_zoo))
        do_solvers = expand_string(do_solvers, list(solvers_zoo))
        for game_name in do_games:
            if not game_name in games_zoo:
                raise ZValueError(f"Cannot find {game_name!r}", available=set(games_zoo))
        for solver_name in do_solvers:
            if not solver_name in solvers_zoo:
                raise ZValueError(f"Cannot find {solver_name!r}", available=set(solvers_zoo))

        for game_name in do_games:
            cgame = context.child(game_name, extra_report_keys=dict(game=game_name))
            game = games_zoo[game_name].game
            rgame = cgame.comp(report_game_visualization, game)
            cgame.add_report(rgame, "game_tree")
            for solver_name in do_solvers:
                c = cgame.child(solver_name, extra_report_keys=dict(solver=solver_name))
                solver_params = solvers_zoo[solver_name].solver_params
                perf_stats = PerformanceStatistics(game_name=game_name, solver_name=solver_name)

                # individual = c.comp(compute_individual_solutions, game, solver_params)
                game_preprocessed = c.comp(preprocess_game, game, solver_params, perf_stats)

                r = c.comp(create_report_preprocessed, game_name, game_preprocessed)
                c.add_report(r, "preprocessed")

                solutions = c.comp(solve_main, game_preprocessed, perf_stats)
                r = c.comp(report_solutions, game_preprocessed, solutions)
                c.add_report(r, "solutions")


def without_compmake(games: Mapping[str, GameSpec], solvers: Mapping[str, SolverSpec]):
    d = "out/tests/"
    for game_name, game_spec in games.items():
        dg = join(d, game_name)
        game = game_spec.game
        r_game = report_game_visualization(game)
        r_game.to_html(join(dg, "report_game.html"))

        for solver_name, solver_spec in solvers.items():
            ds = join(dg, solver_name)
            solver_params = solver_spec.solver_params
            perf_stats = PerformanceStatistics(game_name=game_name, solver_name=solver_name)
            game_preprocessed = preprocess_game(game, solver_params, perf_stats=perf_stats)

            r_preprocessed = create_report_preprocessed(game_name, game_preprocessed)
            r_preprocessed.to_html(join(ds, "r_preprocessed.html"))

            solutions = solve_main(game_preprocessed, perf_stats=perf_stats)
            r_solutions = report_solutions(game_preprocessed, solutions)
            r_solutions.to_html(join(ds, "r_solutions.html"))

            r_perf_stats = report_performance_stats(perf_stats)
            r_perf_stats.to_html(join(ds, "r_perf_stats.html"))


dg_demo = DGDemo.get_sys_main()
