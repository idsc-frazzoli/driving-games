from os.path import join
from itertools import product
from frozendict import frozendict
from parameterized import parameterized

from games import (
    MIX_MNE,
    PURE_STRATEGIES,
    report_solutions,
    create_report_preprocessed,
    UncertaintyParams,
    logger
)
from games.access import get_game_factorization
from games.solve.solution import solve1
from possibilities import PossibilitySet, PossibilityDist
from preferences.preferences_probability import ProbPrefExpectedValue
from preferences import SetPreference1

from factorization.structures import FactorizationSolverParams, FactorizationSolverSpec
from duckie_games.solve import preprocess_duckie_game

from toy_games.n_player_toy_game import get_toy_car_game
from toy_games.n_player_toy_game_zoo import (
    toy_params_star,
    toy_params_x_with_base,
    toy_params_indep_lanes,
    toy_params_two_indep_games,
    toy_params_two_x_joint,
    toy_params_two_x_crossed
)

uncertainty_sets = UncertaintyParams(poss_monad=PossibilitySet(), mpref_builder=SetPreference1)
uncertainty_prob = UncertaintyParams(poss_monad=PossibilityDist(), mpref_builder=ProbPrefExpectedValue)


uncertainty_params = [
    uncertainty_sets,
    # uncertainty_prob,
]

toy_game_params = [
    toy_params_star,
    toy_params_x_with_base,
    toy_params_indep_lanes,
    toy_params_two_indep_games,
    toy_params_two_x_joint,
    toy_params_two_x_crossed
]

params = list(product(toy_game_params, uncertainty_params))


@parameterized(params)
def test_n_player_toy_game(toy_game_parameters, uncert_params):
    """
    N-Player toy game
    """
    d = "out/n_player_toy_car_game/"
    game_name = toy_game_parameters.params_name
    solver_name = "Test"
    game = get_toy_car_game(toy_games_params=toy_game_parameters, uncertainty_params=uncert_params)
    use_factorization = False
    get_factorization = get_game_factorization  # factorization algo used
    dt = toy_game_parameters.dt  # delta-t of discretization
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

    # logger.info(solutions=solutions)

    dg = join(d, game_name)
    ds = join(dg, solver_name)
    r_solutions = report_solutions(game_preprocessed, solutions)
    r_preprocessed = create_report_preprocessed(game_name, game_preprocessed)

    r_solutions.to_html(join(ds, "r_solutions.html"))
    r_preprocessed.to_html(join(ds, "r_preprocessed.html"))
    # logger.info(policies=solutions.game_solution.policies)
    # logger.info(states_to_solutions=solutions.game_solution.states_to_solution)
