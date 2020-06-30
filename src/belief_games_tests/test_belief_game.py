from decimal import Decimal as D

from belief_games import get_leader_follower_game, get_asym
from games import preprocess_game, solve1, SolverParams, STRATEGY_SECURITY, STRATEGY_MIX
from games_scripts.solvers import SolverSpec

def test1():
    print("end!!")
    game_spec = get_asym()
    game1 = game_spec.game
    solver_spec = SolverSpec("test", SolverParams(D(1), STRATEGY_MIX, True))
    game_preprocessed = preprocess_game(game1, solver_spec.solver_params)
    solutions = solve1(game_preprocessed)
