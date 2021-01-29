from decimal import Decimal as D
from os.path import join
from itertools import product
import numpy as np

from parameterized import parameterized

from games import (
    MIX_MNE,
    PURE_STRATEGIES,
    report_solutions,
    create_report_preprocessed,
    PlayerName
)
from games.access import get_game_factorization
from games.solve.solution import solve1


from factorization.structures import FactorizationSolverParams, FactorizationSolverSpec
from factorization.solve import preprocess_game_factorization
from duckie_games.game_generation import get_duckie_game
from duckie_games.solve import preprocess_duckie_game

from duckie_games.zoo import (
    two_player_duckie_game_parameters,
    two_player_duckie_game_parameters_stretched,
    three_player_duckie_game_parameters_stretched,
    uncertainty_prob,
    uncertainty_sets,
)


uncertainty_params = [
    uncertainty_sets,
    # uncertainty_prob,
]

duckie_game_params = [
    # two_player_duckie_game_parameters,
    two_player_duckie_game_parameters_stretched
]

params = list(product(duckie_game_params, uncertainty_params))


@parameterized(params)
def test_two_player_duckie_game(duckie_game_parameters, duckie_uncert_params):
    """
    Similar values as in get_sim() of driving games, but played with duckies
    """
    d = "out/two_player_duckie_game_new_collision_func/"
    game_name = "Two player Duckie game"
    solver_name = "Test"
    game = get_duckie_game(duckie_game_params=duckie_game_parameters, uncertainty_params=duckie_uncert_params)
    use_factorization = False
    get_factorization = get_game_factorization  # factorization algo used
    dt = duckie_game_parameters.dt  # delta-t of discretization
    admissible_strategies = PURE_STRATEGIES
    strategy_multiple_nash = MIX_MNE

    solve_params = FactorizationSolverParams(
            admissible_strategies=admissible_strategies,
            strategy_multiple_nash=strategy_multiple_nash,
            dt=dt,
            use_factorization=use_factorization,
            get_factorization=get_factorization
    )
    solver_spec = FactorizationSolverSpec("test", solve_params)

    game_preprocessed = preprocess_duckie_game(game, solver_spec.solver_params)
    solutions = solve1(game_preprocessed)

    dg = join(d, game_name)
    ds = join(dg, solver_name)
    r_solutions = report_solutions(game_preprocessed, solutions)
    r_preprocessed = create_report_preprocessed(game_name, game_preprocessed)

    r_solutions.to_html(join(ds, "r_solutions.html"))
    r_preprocessed.to_html(join(ds, "r_preprocessed.html"))
    # print(solutions.game_solution.policies)
    # print(solutions.game_solution.states_to_solution)


# uncertainty_params = [
#     uncertainty_sets,
#     # uncertainty_prob,
# ]
#
# duckie_game_params = [
#     three_player_duckie_game_parameters_stretched
# ]
#
# params = list(product(duckie_game_params, uncertainty_params))

# @parameterized(params)
# def test_three_player_duckie_game(duckie_game_parameters, duckie_uncert_params):
#     """
#     Test for a 3 player duckie game
#     """
#     d = "out/three_player_duckie_game/"
#     game_name = "Three player duckie game"
#     solver_name = "Test"
#     game = get_duckie_game(duckie_game_params=duckie_game_parameters, uncertainty_params=duckie_uncert_params)
#     use_factorization = False
#     get_factorization = get_game_factorization  # factorization algo used
#     dt = duckie_game_parameters.dt  # delta-t of discretization
#     admissible_strategies = PURE_STRATEGIES
#     strategy_multiple_nash = MIX_MNE
#
#     solve_params = FactorizationSolverParams(
#         admissible_strategies=admissible_strategies,
#         strategy_multiple_nash=strategy_multiple_nash,
#         dt=dt,
#         use_factorization=use_factorization,
#         get_factorization=get_factorization
#     )
#     solver_spec = FactorizationSolverSpec("test", solve_params)
#
#     game_preprocessed = preprocess_duckie_game(game, solver_spec.solver_params)
#     solutions = solve1(game_preprocessed)
#
#     dg = join(d, game_name)
#     ds = join(dg, solver_name)
#     r_solutions = report_solutions(game_preprocessed, solutions)
#     r_preprocessed = create_report_preprocessed(game_name, game_preprocessed)
#
#     r_solutions.to_html(join(ds, "r_solutions.html"))
#     r_preprocessed.to_html(join(ds, "r_preprocessed.html"))
#     # print(solutions.game_solution.policies)
#     # print(solutions.game_solution.states_to_solution)
