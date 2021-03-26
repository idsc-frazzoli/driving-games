from os.path import join
from typing import Mapping
from reprep import Report

from trajectory_games import (
    StaticTrajectoryGame,
    preprocess_full_game,
    preprocess_player,
    solve_game,
    iterative_best_response,
    StaticSolvingContext,
    report_game_visualization,
    SolvedStaticTrajectoryGame,
    report_nash_eq,
    report_preferences,
    get_trajectory_game,
)

plot_gif = True                 # gif vs image for viz
only_traj = False               # Only trajectory generation vs full game
filename = "r_game_all1.html"


def create_reports(game: StaticTrajectoryGame, nash_eq: Mapping[str, SolvedStaticTrajectoryGame], folder: str):
    d = "out/tests/"
    if not only_traj:
        print(
            f"Weak = {len(nash_eq['weak'])}, "
            f"Indiff = {len(nash_eq['indiff'])}, "
            f"Incomp = {len(nash_eq['incomp'])}, "
            f"Strong = {len(nash_eq['strong'])}."
        )
    r_game = Report()
    r_game.add_child(report_game_visualization(game=game))
    if not only_traj:
        r_game.add_child(report_nash_eq(game=game, nash_eq=nash_eq, plot_gif=plot_gif))
        r_game.add_child(report_preferences(game=game))
    r_game.to_html(join(d, folder + filename))

    from world import LaneSegmentHashable
    from trajectory_games.metrics import CollisionEnergy, MinimumClearance
    print(f"LanePose time = {LaneSegmentHashable.time:.2f} s")
    coll, clear = CollisionEnergy(), MinimumClearance()
    print(f"Collision time = {coll.time:.2f} s")
    print(f"Clearance time = {clear.time:.2f} s")


def test_trajectory_game():
    game: StaticTrajectoryGame = get_trajectory_game()
    context: StaticSolvingContext = preprocess_full_game(sgame=game, only_traj=only_traj)

    if only_traj:
        nash_eq = {}
    else:
        nash_eq: Mapping[str, SolvedStaticTrajectoryGame] = solve_game(context=context)
    create_reports(game=game, nash_eq=nash_eq, folder="brute_force/")


def test_trajectory_game_best_response():
    n_runs = 100      # Number of random runs for best response

    game: StaticTrajectoryGame = get_trajectory_game()
    context: StaticSolvingContext = preprocess_player(sgame=game, only_traj=only_traj)

    if only_traj:
        nash_eq = {}
    else:
        nash_eq: Mapping[str, SolvedStaticTrajectoryGame] = \
            iterative_best_response(context=context, n_runs=n_runs)
    create_reports(game=game, nash_eq=nash_eq, folder="best_response/")
