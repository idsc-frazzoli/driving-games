from typing import List, Mapping

from decent_params import DecentParams
from quickapp import QuickApp, QuickAppContext

from games import SolvingContext
from trajectory_games import get_trajectory_game, preprocess_full_game, TrajectoryGame, Solution, SolvedTrajectoryGame


def bruteforce_preprocess(config_str: str) -> (TrajectoryGame, SolvingContext):
    game: TrajectoryGame = get_trajectory_game(config_str=config_str)
    solve_context = preprocess_full_game(sgame=game, only_traj=False)
    return game, solve_context


def bruteforce_solve(context: SolvingContext) -> Mapping[str, SolvedTrajectoryGame]:
    sol = Solution()
    return sol.solve_game(context=context)


def bruteforce_viz(game: TrajectoryGame, nash_eq=Mapping[str, SolvedTrajectoryGame]):
    game.game_vis.init_plot_dict(values=nash_eq["weak"])


class CrashingExperiments(QuickApp):
    """ Main Experiments runner """

    def define_options(self, params: DecentParams):
        pass

    def define_jobs_context(self, context: QuickAppContext):
        bruteforce: List[str] = ["basic", "basic2"]

        for exp in bruteforce:
            cexp = context.child(exp, extra_report_keys=dict(experiment=exp))
            game, solving_context = cexp.comp(bruteforce_preprocess, exp)
            nash_eq = cexp.comp(bruteforce_solve, solving_context)
            report = cexp.comp(bruteforce_viz, game, nash_eq=nash_eq)
            cexp.add_report(report, "something")
