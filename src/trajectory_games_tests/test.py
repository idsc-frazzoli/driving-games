from os.path import join

from reprep import Report

from trajectory_games import TrajectoryGame, compute_solving_context, solve_game, StaticSolvingContext, report_game_visualization

from trajectory_games.game_factory import get_trajectory_game


def test_trajectory_game():
    d = "out/tests/"
    game: TrajectoryGame = get_trajectory_game()
    context: StaticSolvingContext = compute_solving_context(sgame=game)

    nash_eq = solve_game(context=context)
    print(
        f"Weak = {len(nash_eq['weak_nash'])}, "
        f"Indiff = {len(nash_eq['indiff_nash'])}, "
        f"Incomp = {len(nash_eq['incomp_nash'])}, "
        f"Strong = {len(nash_eq['strong_nash'])}."
    )

    r_game: Report = report_game_visualization(game=game)
    r_game.to_html(join(d, "r_animation.html"))
    # r_game.to_html(join(d, "r_animation.r_game"))

    a = 2
