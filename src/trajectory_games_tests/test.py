from trajectory_games import TrajectoryGame, compute_solving_context, solve_game, StaticSolvingContext

from trajectory_games.game_factory import get_trajectory_game


def test_trajectory_game():
    game: TrajectoryGame = get_trajectory_game()
    context: StaticSolvingContext = compute_solving_context(sgame=game)

    nash_eq = solve_game(context=context)
    print(
        f"Weak = {len(nash_eq['weak_nash'])}, "
        f"Indiff = {len(nash_eq['indiff_nash'])}, "
        f"Incomp = {len(nash_eq['incomp_nash'])}, "
        f"Strong = {len(nash_eq['strong_nash'])}."
    )
    a = 2
