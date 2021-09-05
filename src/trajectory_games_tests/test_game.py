import os
from copy import deepcopy
from os.path import join
from typing import Mapping, Dict, Optional

from reprep import Report
from yaml import safe_load

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
    get_leader_follower_game,
    PosetalPreference,
    solve_leader_follower,
    report_leader_follower_solution,
    solve_recursive_game,
    report_leader_follower_recursive,
    config_dir,
    VehicleState,
)
from trajectory_games.trajectory_game import LeaderFollowerGame, LeaderFollowerGameSolvingContext

plot_gif = False  # gif vs image for viz
only_traj = False  # Only trajectory generation vs full game
d = "out/tests/"
filename = "r_game_all.html"


def create_reports(game: TrajectoryGame, nash_eq: Mapping[str, SolvedTrajectoryGame],
                   r_game: Report, gif: bool = plot_gif):
    if not only_traj:
        print(", ".join(f"{k.capitalize()} = {len(v)}" for k, v in nash_eq.items()))
        r_game.add_child(report_nash_eq(game=game, nash_eq=nash_eq, plot_gif=gif))


def report_single(game: TrajectoryGame, nash_eq: Mapping[str, SolvedTrajectoryGame], folder: str):
    r_game = Report()
    r_game.add_child(report_game_visualization(game=game))
    create_reports(game=game, nash_eq=nash_eq, r_game=r_game)
    prefs = {p.name: p.preference for p in game.game_players.values()}
    r_game.add_child(report_preferences(viz=game.game_vis, players=prefs))
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
    game: TrajectoryGame = get_trajectory_game(config_str="basic")
    context: SolvingContext = preprocess_full_game(sgame=game, only_traj=only_traj)

    if only_traj:
        nash_eq = {}
    else:
        sol = Solution()
        nash_eq: Mapping[str, SolvedTrajectoryGame] = sol.solve_game(context=context)
        game.game_vis.init_plot_dict(values=nash_eq["weak"])
    report_single(game=game, nash_eq=nash_eq, folder=folder)


def test_trajectory_game_best_response():
    folder = "best_response/"
    n_runs = 100  # Number of random runs for best response

    game: TrajectoryGame = get_trajectory_game(config_str="basic")
    context: SolvingContext = preprocess_player(sgame=game, only_traj=only_traj)

    if only_traj:
        nash_eq = {}
    else:
        nash_eq: Mapping[str, SolvedTrajectoryGame] = \
            iterative_best_response(context=context, n_runs=n_runs)
        game.game_vis.init_plot_dict(values=nash_eq["weak"])
    report_single(game=game, nash_eq=nash_eq, folder=folder)


def test_trajectory_game_lexi():
    folder = "lexi/"

    players_file = os.path.join(config_dir, "players.yaml")
    with open(players_file) as load_file:
        config = safe_load(load_file)["lexi"]
    states = config["states"]
    prefs = config["prefs"]

    game: TrajectoryGame = get_trajectory_game(config_str="lexi")
    report = Report()

    pname = next(iter(game.game_players.keys()))
    player = game.game_players[pname]
    for i in range(len(states)):
        player.state = game.ps.unit(VehicleState.from_config(name=states[i]))
        for j in range(len(prefs)):
            try:
                player.preference = PosetalPreference(pref_str=prefs[j], use_cache=False)
            except:
                a = 2

            context: SolvingContext = preprocess_player(sgame=game, only_traj=only_traj)
            nash_eq: Mapping[str, SolvedTrajectoryGame] = \
                iterative_best_response(context=context, n_runs=1)
            game.game_vis.init_plot_dict(values=nash_eq["weak"])
            r_game = Report(f"State={i + 1}, Pref={j + 1}")
            r_game.add_child(report_game_visualization(game=game))
            create_reports(game=game, nash_eq=nash_eq, r_game=r_game)
            player_prefs = {p.name: p.preference for p in game.game_players.values()}
            r_game.add_child(report_preferences(viz=game.game_vis, players=player_prefs))
            report.add_child(r_game)
    report.to_html(join(d, folder + filename))


def test_trajectory_game_levels():
    folder = "levels_cases/"
    pref = "pref_level"

    game = get_trajectory_game(config_str="basic")
    sol = Solution()

    def update_prefs(suffix: str):
        for pname, player in game.game_players.items():
            player.preference = PosetalPreference(pref_str=f"{pref}_{suffix}", use_cache=False)

    r_levels: Dict[int, Report] = {}

    def play_stage(stage: int) -> Mapping[str, SolvedTrajectoryGame]:
        print(f"\nLevel = {stage}")
        name = str(stage)
        update_prefs(suffix=name)
        context = preprocess_full_game(sgame=game, only_traj=only_traj)
        stage_eq = sol.solve_game(context=context, cache_dom=True)
        if game.game_vis.plot_dict is None and len(stage_eq["weak"]) <= 200:
            game.game_vis.init_plot_dict(values=stage_eq["weak"])
        if stage not in r_levels:
            rep = Report()
            create_reports(game=game, nash_eq=stage_eq, r_game=rep, gif=plot_gif)
            node: Report = rep.last()
            node.nid = f"Pref_{name}"
            r_levels[stage] = node
        return stage_eq

    nash_eqf: Optional[Mapping[str, SolvedTrajectoryGame]] = None
    for i in range(1, 5):
        nash_eqf = play_stage(stage=i + 1)

    r_game = Report()
    r_game.add_child(report_game_visualization(game=game))
    create_reports(game=game, nash_eq=nash_eqf, r_game=r_game, gif=plot_gif)
    prefs = {p.name: p.preference for p in game.game_players.values()}
    r_game.add_child(report_preferences(viz=game.game_vis, players=prefs))
    for level in r_levels.values():
        r_game.add_child(level)
    r_game.to_html(join(d, folder + filename))
    report_times()


def test_leader_follower():
    folder = "LF/"
    game: LeaderFollowerGame = get_leader_follower_game()
    context: SolvingContext = preprocess_full_game(sgame=game, only_traj=False)
    assert isinstance(context, LeaderFollowerGameSolvingContext)
    solutions = solve_leader_follower(context=context)
    r_game = Report()
    r_game.add_child(report_game_visualization(game=game))
    r_game.add_child(report_leader_follower_solution(game=game, solution=solutions, plot_gif=plot_gif))
    r_game.to_html(join(d, folder + filename))


def test_leader_follower_recursive():
    folder = "LF_recursive/"
    game: LeaderFollowerGame = get_leader_follower_game()
    game_init = deepcopy(game)
    result = solve_recursive_game(game=game)
    r_game = Report()
    r_game.add_child(report_game_visualization(game=game_init))
    r_game.add_child(report_leader_follower_recursive(game=game_init, result=result, plot_gif=plot_gif))
    r_game.to_html(join(d, folder + filename))


if __name__ == '__main__':
    d = 'trajectory_games_tests/' + d
    # test_trajectory_game_brute_force()
    # test_trajectory_game_best_response()
    test_trajectory_game_levels()
