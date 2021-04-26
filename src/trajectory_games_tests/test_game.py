from os.path import join
from typing import Mapping, List
from reprep import Report

from trajectory_games import (
    TrajectoryGame,
    preprocess_full_game,
    preprocess_player,
    Solution,
    iterative_best_response,
    SolvingContext,
    report_game_visualization,
    SolvedTrajectoryGame,
    report_nash_eq,
    report_preferences,
    get_trajectory_game,
    PosetalPreference,
)

plot_gif = True                 # gif vs image for viz
only_traj = False               # Only trajectory generation vs full game
d = "out/tests/"
filename = "r_game_all.html"


def create_reports(game: TrajectoryGame, nash_eq: Mapping[str, SolvedTrajectoryGame],
                   r_game: Report, gif: bool = plot_gif):
    if not only_traj:
        print(
            f"Weak = {len(nash_eq['weak'])}, "
            f"Indiff = {len(nash_eq['indiff'])}, "
            f"Incomp = {len(nash_eq['incomp'])}, "
            f"Strong = {len(nash_eq['strong'])}."
        )
        r_game.add_child(report_nash_eq(game=game, nash_eq=nash_eq, plot_gif=gif))


def report_single(game: TrajectoryGame, nash_eq: Mapping[str, SolvedTrajectoryGame], folder: str):
    r_game = Report()
    r_game.add_child(report_game_visualization(game=game))
    create_reports(game=game, nash_eq=nash_eq, r_game=r_game)
    r_game.add_child(report_preferences(game=game))
    r_game.to_html(join(d, folder + filename))
    report_times()


def report_times():
    from world import LaneSegmentHashable
    from trajectory_games.metrics import CollisionEnergy, MinimumClearance
    print(f"LanePose time = {LaneSegmentHashable.time:.2f} s")
    coll, clear = CollisionEnergy(), MinimumClearance()
    print(f"Collision time = {coll.time:.2f} s")
    print(f"Clearance time = {clear.time:.2f} s")


def test_trajectory_game_brute_force():
    folder = "brute_force/"
    game: TrajectoryGame = get_trajectory_game()
    context: SolvingContext = preprocess_full_game(sgame=game, only_traj=only_traj)

    if only_traj:
        nash_eq = {}
    else:
        sol = Solution()
        nash_eq: Mapping[str, SolvedTrajectoryGame] = sol.solve_game(context=context)
    report_single(game=game, nash_eq=nash_eq, folder=folder)


def test_trajectory_game_best_response():
    folder = "best_response/"
    n_runs = 100      # Number of random runs for best response

    game: TrajectoryGame = get_trajectory_game()
    context: SolvingContext = preprocess_player(sgame=game, only_traj=only_traj)

    if only_traj:
        nash_eq = {}
    else:
        nash_eq: Mapping[str, SolvedTrajectoryGame] = \
            iterative_best_response(context=context, n_runs=n_runs)
    report_single(game=game, nash_eq=nash_eq, folder=folder)


def test_trajectory_game_levels():
    folder = "levels/"
    pref = "pref_d7"

    game = get_trajectory_game()
    context: SolvingContext
    nash_eq: Mapping[str, SolvedTrajectoryGame] = {}
    sol = Solution()

    def update_prefs(level: int):
        for pname, player in game.game_players.items():
            player.preference = PosetalPreference(pref_str=f"{pref}_{level}", use_cache=False)

    r_levels: List[Report] = []
    for i in range(2, 8):
        print(f"\nLevel = {i}")
        update_prefs(level=i)
        context = preprocess_full_game(sgame=game, only_traj=only_traj)
        cache_dom = i <= 3  # Preferences are not extra levels of previous after this!
        nash_eq = sol.solve_game(context=context, cache_dom=cache_dom)
        rep = Report()
        create_reports(game=game, nash_eq=nash_eq, r_game=rep, gif=False)
        node: Report = rep.last()
        node.nid = f"Pref_{i}"
        r_levels.append(node)

    r_game = Report()
    r_game.add_child(report_game_visualization(game=game))
    create_reports(game=game, nash_eq=nash_eq, r_game=r_game, gif=True)
    r_game.add_child(report_preferences(game=game))
    for level in r_levels:
        r_game.add_child(level)
    r_game.to_html(join(d, folder + filename))
    report_times()


if __name__ == '__main__':
    d = 'trajectory_games_tests/' + d
    # test_trajectory_game_brute_force()
    # test_trajectory_game_best_response()
    test_trajectory_game_levels()
