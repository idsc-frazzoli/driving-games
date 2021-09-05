from typing import List, Mapping

from decent_params import DecentParams
from quickapp import QuickApp, QuickAppContext
from reprep import Report

from trajectory_games import get_trajectory_game, preprocess_full_game, TrajectoryGame, Solution, SolvedTrajectoryGame, \
    report_preferences, SolvingContext, report_game_visualization
from trajectory_games_tests.test_game import create_reports


def bruteforce_solve(context: SolvingContext) -> Mapping[str, SolvedTrajectoryGame]:
    sol = Solution()
    return sol.solve_game(context=context)


def bruteforce_report(game: TrajectoryGame, nash_eq=Mapping[str, SolvedTrajectoryGame]) -> Report:
    game.game_vis.init_plot_dict(values=nash_eq["weak"])
    r_game = Report()
    print(game.game_vis)
    r_game.add_child(report_game_visualization(game=game))
    create_reports(game=game, nash_eq=nash_eq, r_game=r_game, gif=True)
    prefs = {p.name: p.preference for p in game.game_players.values()}
    r_game.add_child(report_preferences(viz=game.game_vis, players=prefs))
    return r_game


class RalExperiments(QuickApp):
    """ Main Experiments runner """

    def define_options(self, params: DecentParams):
        pass

    def define_jobs_context(self, context: QuickAppContext):
        bruteforce: List[str] = ["basic", ]  # "basic2"]

        for exp in bruteforce:
            cexp = context.child(exp, extra_report_keys=dict(experiment=exp))
            game = cexp.comp(get_trajectory_game, exp)
            solving_context = cexp.comp(preprocess_full_game, game)
            nash_eq = cexp.comp(bruteforce_solve, solving_context)
            report = cexp.comp(bruteforce_report, game, nash_eq=nash_eq)
            cexp.add_report(report, "")


run_ral_exp = RalExperiments.get_sys_main()
