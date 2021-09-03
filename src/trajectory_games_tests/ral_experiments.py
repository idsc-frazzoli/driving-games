from typing import List, Mapping

from decent_params import DecentParams
from quickapp import QuickApp, QuickAppContext

from games import SolvingContext
from trajectory_games import get_trajectory_game, preprocess_full_game, TrajectoryGame, Solution, SolvedTrajectoryGame


def bruteforce_solve(context: SolvingContext) -> Mapping[str, SolvedTrajectoryGame]:
    sol = Solution()
    return sol.solve_game(context=context)


def bruteforce_viz(game: TrajectoryGame, nash_eq=Mapping[str, SolvedTrajectoryGame]):
    game.game_vis.init_plot_dict(values=nash_eq["weak"])


class RalExperiments(QuickApp):
    """ Main Experiments runner """

    def define_options(self, params: DecentParams):
        pass

    def define_jobs_context(self, context: QuickAppContext):
        bruteforce: List[str] = ["basic", "basic2"]

        for exp in bruteforce:
            cexp = context.child(exp, extra_report_keys=dict(experiment=exp))
            game = cexp.comp(get_trajectory_game, exp)
            solving_context = cexp.comp(preprocess_full_game, game)
            nash_eq = cexp.comp(bruteforce_solve, solving_context)
            report = cexp.comp(bruteforce_viz, game, nash_eq=nash_eq)
            cexp.add_report(report, "something")


run_ral_exp = RalExperiments.get_sys_main()
