from games import STRATEGY_MIX, STRATEGY_SECURITY, preprocess_game, solve1
from games_scripts import solvers_zoo, games_zoo


def test_single_game():
    strategies = [STRATEGY_MIX, STRATEGY_SECURITY]

    solver_spec = solvers_zoo["solver-1-" + strategies[0]]
    solver_params = solver_spec.solver_params
    game_spec = games_zoo["sym_v1"]
    game_preprocessed = preprocess_game(game_spec.game, solver_params)
    solutions = solve1(game_preprocessed)


if __name__ == "__main__":
    test_single_game()
