__all__ = ["dg_demo"]

from decimal import Decimal as D
from typing import Dict

from decent_params import DecentParams
from quickapp import QuickApp, QuickAppContext
from .access import preprocess_game
from .driving_example import get_game1
from .game_def import Game
from .reports import create_report_preprocessed
from .solution import solve1, SolverParams


class App(QuickApp):
    """ Main function """

    def define_options(self, params: DecentParams):
        params.add_string("games", default="game1")
        params.add_string("solvers", default="solver1")

    def define_jobs_context(self, context: QuickAppContext):
        # Creates the examples
        games: Dict[str, Game] = {}
        games["game1"] = get_game1()

        # The solution parameters
        solvers = {
            "solver1": SolverParams(D(1.0)),
            "solver0.5": SolverParams(D(0.5))
        }

        do_solvers = self.get_options().solvers.split(',')
        do_games = self.get_options().games.split(',')
        for game_name in do_games:
            cgame = context.child(game_name)
            game = games[game_name]

            for solver_name in do_solvers:
                cgamesolver = cgame.child(solver_name)
                solver = solvers[solver_name]
                game_preprocessed = cgamesolver.comp(preprocess_game, game, solver.dt)

                r = cgamesolver.comp(create_report_preprocessed, game_name, game_preprocessed)
                cgamesolver.add_report(
                    r, "report_preprocessed", game_name=game_name, solver_name=solver_name,
                )

                cgamesolver.comp(solve1, game_preprocessed)
            #
            # # create solution
            # solution = context.comp(solve, city, sp)
            # # create report


dg_demo = App.get_sys_main()
