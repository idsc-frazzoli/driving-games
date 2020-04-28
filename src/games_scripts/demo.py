from decent_params import DecentParams
from quickapp import QuickApp, QuickAppContext

from games import solve1
from games.access import preprocess_game
from games.animations import report_animation
from games.reports import create_report_preprocessed, report_game_visualization
from games.solution import solve_random
from .solvers import solvers_zoo
from .zoo import games_zoo

__all__ = ["dg_demo", "DGDemo"]


class DGDemo(QuickApp):
    """ Main function """

    def define_options(self, params: DecentParams):
        params.add_string("games", default="game1")
        params.add_string("solvers", default="solver1")

    def define_jobs_context(self, context: QuickAppContext):

        do_solvers = self.get_options().solvers.split(",")
        do_games = self.get_options().games.split(",")
        for game_name in do_games:
            cgame = context.child(game_name)
            game = games_zoo[game_name].game
            rgame = cgame.comp(report_game_visualization, game)
            cgame.add_report(
                rgame, "visualization", game_name=game_name,
            )
            for solver_name in do_solvers:
                cgamesolver = cgame.child(solver_name)
                solver = solvers_zoo[solver_name].solver_params

                game_preprocessed = cgamesolver.comp(preprocess_game, game, solver.dt)

                r = cgamesolver.comp(create_report_preprocessed, game_name, game_preprocessed)
                cgamesolver.add_report(
                    r, "report_preprocessed", game_name=game_name, solver_name=solver_name,
                )

                cgamesolver.comp(solve1, game_preprocessed)
                random_sim = cgamesolver.comp(solve_random, game_preprocessed)
                r = cgamesolver.comp(report_animation, game_preprocessed, random_sim)
                cgamesolver.add_report(
                    r, "random_animation", game_name=game_name, solver_name=solver_name,
                )


dg_demo = DGDemo.get_sys_main()
