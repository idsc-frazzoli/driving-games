from trajectory_game import (
    TrajectoryGame,
    compute_solving_context,
    solve_game,
    StaticSolvingContext,
    SolvedTrajectoryGame,
)
from trajectory_game.game_factory import get_trajectory_game


def test_trajectory_game():
    game: TrajectoryGame = get_trajectory_game()
    context: StaticSolvingContext = compute_solving_context(sgame=game)

    indiff_nash: SolvedTrajectoryGame
    incomp_nash: SolvedTrajectoryGame
    weak_nash: SolvedTrajectoryGame
    strong_nash: SolvedTrajectoryGame

    indiff_nash, incomp_nash, weak_nash, strong_nash = solve_game(context=context)

    a = 2
