from games import STRATEGY_MIX, STRATEGY_SECURITY, preprocess_game, solve1
from games_scripts import solvers_zoo
from handcrafted_games.handcrafted_game import get_handcrafted_game_spec


def test_run_handcrafted_games():
    strategies = [STRATEGY_MIX, STRATEGY_SECURITY]
    for strategy in strategies:
        solver_spec = solvers_zoo["solver-1-"+strategy]
        solver_params = solver_spec.solver_params
        game_spec = get_handcrafted_game_spec()
        game_preprocessed = preprocess_game(game_spec.game, solver_params)
        solutions = solve1(game_preprocessed)
        # todo check solutions with what we expect
