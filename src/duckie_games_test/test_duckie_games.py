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

from duckie_games_test.test_game_generation import duckie_vehicle_parameters, duckie_game_parameters, uncertainty_prob, uncertainty_sets, DuckieGameParams


uncertainty_params = [
    uncertainty_sets,
    #uncertainty_prob,
]
vehicle_params = [duckie_vehicle_parameters]

duckie_game_params = [duckie_game_parameters]

params = list(product(vehicle_params, duckie_game_params, uncertainty_params))


@parameterized(params)
def test_two_player_duckie_game(duckie_vehicle_parameters, duckie_game_parameters, duckie_uncert_params):
    d = "out/two_player_duckie_game/"
    game_name = "Two player Duckie Game"
    solver_name = "Test"
    game = get_duckie_game(
        vehicles_params=duckie_vehicle_parameters,
        game_params=duckie_game_parameters,
        uncertainty_params=duckie_uncert_params
    )
    use_factorization = True
    get_factorization = get_game_factorization  # factorization algo used
    dt = D(1)  # delta-t of discretization
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


player_numbers = 3
player_names = [PlayerName(f"Duckie_{_ + 1}") for _ in range(0, player_numbers)]
player_bottom = player_numbers // 2
player_side = player_numbers - player_bottom
spacing_x = [int(_) for _ in np.linspace(0, 22, player_bottom + 2)[1:-1]]
spacing_y = [int(_) for _ in np.linspace(0, 22, player_side + 2)[1:-1]]
poses = (
    [(D(spacing_x[_]), D(0), D(+90)) if _ < player_bottom
     else (D(22), D(spacing_y[_ - player_bottom]), D(-180))
     for _ in range(player_numbers)]
)
initial_poses = {player_names[_]: poses[_] for _ in range(0, player_numbers)}

duckie_game_params_n_player = [
    DuckieGameParams(
        player_number=player_numbers,
        player_names=player_names,
        initial_poses=initial_poses
    )
]

params_n_player = list(product(vehicle_params, duckie_game_params_n_player, uncertainty_params))


# todo make game ready for more than 2 players
@parameterized(params_n_player)
def test_n_player_duckie_game(duckie_vehicle_parameters, duckie_game_parameters, duckie_uncert_params):
    d = f"out/duckie_game_{player_numbers}_player/"
    game_name = f"{player_numbers} Player Duckie Game"
    solver_name = f"Test {player_numbers}-players"
    game = get_duckie_game(
        vehicles_params=duckie_vehicle_parameters,
        game_params=duckie_game_parameters,
        uncertainty_params=duckie_uncert_params
    )
    use_factorization = True
    get_factorization = get_game_factorization  # factorization algo used
    dt = D(1)  # delta-t of discretization
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
