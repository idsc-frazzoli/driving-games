from os.path import join
from typing import Mapping
from reprep import Report

from trajectory_games import (
    TrajectoryGame,
    preprocess_full_game,
    preprocess_player,
    solve_game,
    iterative_best_response,
    StaticSolvingContext,
    report_game_visualization,
    SolvedTrajectoryGame,
    report_nash_eq,
    report_preferences,
    get_trajectory_game,
)

plot_gif = True                 # gif vs image for viz
only_traj = False               # Only trajectory generation vs full game
filename = "r_game_all.html"


def create_reports(game: TrajectoryGame, nash_eq: Mapping[str, SolvedTrajectoryGame], folder: str):
    d = "out/tests/"
    if not only_traj:
        print(
            f"Weak = {len(nash_eq['weak_nash'])}, "
            f"Indiff = {len(nash_eq['indiff_nash'])}, "
            f"Incomp = {len(nash_eq['incomp_nash'])}, "
            f"Strong = {len(nash_eq['strong_nash'])}."
        )
    r_game = Report()
    r_game.add_child(report_game_visualization(game=game))
    if not only_traj:
        r_game.add_child(report_nash_eq(game=game, nash_eq=nash_eq, plot_gif=plot_gif))
        r_game.add_child(report_preferences(game=game))
    r_game.to_html(join(d, folder + filename))

    from world import LaneSegmentHashable
    print(f"LanePose time = {LaneSegmentHashable.time:.2f} s")


def test_trajectory_game():
    game: TrajectoryGame = get_trajectory_game()
    context: StaticSolvingContext = preprocess_full_game(sgame=game, only_traj=only_traj)

    nash_eq: Mapping[str, SolvedTrajectoryGame] = solve_game(context=context)
    create_reports(game=game, nash_eq=nash_eq, folder="brute_force/")


def test_trajectory_game_best_response():
    n_runs = 50      # Number of random runs for best response

    game: TrajectoryGame = get_trajectory_game()
    context: StaticSolvingContext = preprocess_player(sgame=game, only_traj=only_traj)

    if only_traj:
        nash_eq = {}
    else:
        nash_eq: Mapping[str, SolvedTrajectoryGame] = \
            iterative_best_response(context=context, n_runs=n_runs)
    create_reports(game=game, nash_eq=nash_eq, folder="best_response/")
