from os.path import join
from typing import Mapping
from reprep import Report

from trajectory_games import TrajectoryGame, compute_solving_context, solve_game, StaticSolvingContext, \
    report_game_visualization, SolvedTrajectoryGame, report_nash_eq, \
    report_preferences, get_trajectory_game


def test_trajectory_game():
    d = "out/tests/"
    game: TrajectoryGame = get_trajectory_game()
    context: StaticSolvingContext = compute_solving_context(sgame=game)

    nash_eq: Mapping[str, SolvedTrajectoryGame] = solve_game(context=context)
    print(
        f"Weak = {len(nash_eq['weak_nash'])}, "
        f"Indiff = {len(nash_eq['indiff_nash'])}, "
        f"Incomp = {len(nash_eq['incomp_nash'])}, "
        f"Strong = {len(nash_eq['strong_nash'])}."
    )

    r_game = Report()
    r_game.add_child(report_game_visualization(game=game))
    r_game.add_child(report_nash_eq(game=game, nash_eq=nash_eq))
    r_game.add_child(report_preferences(game=game))
    r_game.to_html(join(d, "r_animation.html"))

    a = 2
