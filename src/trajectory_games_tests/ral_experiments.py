from typing import List, Mapping

from decent_params import DecentParams
from quickapp import QuickApp, QuickAppContext
from reprep import Report

from trajectory_games import get_trajectory_game, preprocess_full_game, TrajectoryGame, Solution, SolvedTrajectoryGame, \
    report_preferences, SolvingContext, report_game_visualization
from trajectory_games_tests.test_game import create_reports


def bruteforce_solve(context: SolvingContext) -> (Mapping[str, SolvedTrajectoryGame], Solution):
    sol = Solution()
    return sol.solve_game(context=context, cache_dom=True)


def bruteforce_report(game: TrajectoryGame, nash_eq=Mapping[str, SolvedTrajectoryGame],
                      gif: bool = False) -> Report:
    game.game_vis.init_plot_dict(values=nash_eq["weak"])
    r_game = Report()
    if gif:  # first level we plot everything
        r_game.add_child(report_game_visualization(game=game))
    create_reports(game=game, nash_eq=nash_eq, r_game=r_game, gif=gif)
    prefs = {p.name: p.preference for p in game.game_players.values()}
    r_game.add_child(report_preferences(viz=game.game_vis, players=prefs))
    return r_game


class RalExperiments(QuickApp):
    """ Main Experiments runner """

    def define_options(self, params: DecentParams):
        pass

    def define_jobs_context(self, context: QuickAppContext):
        bruteforce: List[str] = ["ral_19", "ral_20"]  # "ral_06",

        for exp in bruteforce:
            cexp = context.child(exp, extra_report_keys=dict(experiment=exp))

            for level in range(3):
                pref_str = f"{exp}_level_{level}"
                game = cexp.comp(get_trajectory_game, pref_str)
                solving_context = cexp.comp(preprocess_full_game, game)
                nash_eq = cexp.comp(bruteforce_solve, solving_context)
                gif = True if level == 0 else False
                report = cexp.comp(bruteforce_report, game, nash_eq=nash_eq, gif=gif)
                cexp.add_report(report, f"refinement{level}")


run_ral_exp = RalExperiments.get_sys_main()
