from os.path import join
from typing import Mapping

from decent_params import DecentParams
from games import (
    create_report_preprocessed,
    preprocess_game,
    report_game_visualization,
    report_solutions,
    solve1,
)
from games import GameSpec
from quickapp import QuickApp, QuickAppContext
from zuper_commons.text import expand_string
from zuper_commons.types import ZValueError
from driving_games.solvers import solvers_zoo, SolverSpec
from games_zoo.zoo import games_zoo

__all__ = ["dg_demo", "DGDemo", "without_compmake"]


class DGDemo(QuickApp):
    """Main function"""

    def define_options(self, params: DecentParams):
        params.add_string("games", default="asym_v1_sets")
        params.add_string("solvers", default="solver-1-security-fact")

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
                # individual = c.comp(compute_individual_solutions, game, solver_params)
                game_preprocessed = c.comp(preprocess_game, game, solver_params)

                r = c.comp(create_report_preprocessed, game_name, game_preprocessed)
                c.add_report(r, "preprocessed")

                solutions = c.comp(solve1, game_preprocessed)
                r = c.comp(report_solutions, game_preprocessed, solutions)
                c.add_report(r, "solutions")


def without_compmake(games: Mapping[str, GameSpec], solvers: Mapping[str, SolverSpec]):
    d = "out/tests/"
    for game_name, game_spec in games.items():
        dg = join(d, game_name)
        game = game_spec.game
        r_game = report_game_visualization(game)
        r_game.to_html(join(dg, "r_animation.r_game"))

        for solver_name, solver_spec in solvers.items():
            ds = join(dg, solver_name)
            solver_params = solver_spec.solver_params
            game_preprocessed = preprocess_game(game, solver_params)
            solutions = solve1(game_preprocessed)
            # random_sim = solve_random(game_preprocessed)

            r_solutions = report_solutions(game_preprocessed, solutions)
            r_preprocessed = create_report_preprocessed(game_name, game_preprocessed)

            r_solutions.to_html(join(ds, "r_solutions.html"))
            r_preprocessed.to_html(join(ds, "r_preprocessed.html"))


dg_demo = DGDemo.get_sys_main()
