from decimal import Decimal as D
from os.path import join
from itertools import product
import numpy as np

from parameterized import parameterized

from games import (
    MIX_MNE,
    SECURITY_MNE,
    MIX_STRATEGIES,
    PURE_STRATEGIES,
    report_solutions,
    create_report_preprocessed,
    UncertaintyParams,
    logger
)

from games.access import get_game_factorization
from games.solve.solution import solve1


from factorization.structures import FactorizationSolverParams, FactorizationSolverSpec
from factorization.solve import preprocess_game_factorization
from duckie_games.game_generation import get_duckie_game
from duckie_games.solve import preprocess_duckie_game

from duckie_games.zoo import (
    two_player_4way,
    two_player_4way_intersection_only,
    three_player_4way,
    three_player_4way_intersection_only,
    three_player_4way_double,
    three_player_4way_double_intersection_only,
    uncertainty_prob,
    uncertainty_sets,
)


uncertainty_params = [
    uncertainty_sets,
    # uncertainty_prob,
]

duckie_game_params = [
    two_player_4way,
    # two_player_4way_intersection_only,
    # three_player_4way,
    # three_player_4way_intersection_only,
    # three_player_4way_double,
    # three_player_4way_double_intersection_only,
]

strategies = [
    PURE_STRATEGIES,
    # MIX_STRATEGIES
]

nash_strategy = [
    MIX_MNE,
    # SECURITY_MNE
]

use_factorization = [
    # True,
    False
]
params = list(product(duckie_game_params, uncertainty_params, strategies, nash_strategy, use_factorization))


@parameterized(params)
def test_duckie_games(duckie_game_parameters, duckie_uncert_params, strat, nash_strat, use_fact):
    """
    n-player duckie game tests
    """
    player_number = duckie_game_parameters.player_number
    m = duckie_game_parameters.map_name
    logger.info(f"Starting test: {duckie_game_parameters.desc}")
    d = "out/"
    game_name = f"{player_number}_player_{m}/"

    solver_name = f"{strat}-{nash_strat}{'-fact' if use_fact else ''}"
    game = get_duckie_game(duckie_game_params=duckie_game_parameters, uncertainty_params=duckie_uncert_params)
    use_factorization = use_fact
    get_factorization = get_game_factorization  # factorization algo used
    dt = duckie_game_parameters.dt  # delta-t of discretization
    admissible_strategies = strat
    strategy_multiple_nash = nash_strat

    solve_params = FactorizationSolverParams(
            admissible_strategies=admissible_strategies,
            strategy_multiple_nash=strategy_multiple_nash,
            dt=dt,
            use_factorization=use_factorization,
            get_factorization=get_factorization
    )
    solver_spec = FactorizationSolverSpec(solver_name, solve_params)

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
