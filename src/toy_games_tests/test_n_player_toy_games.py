from os.path import join
from itertools import product
from parameterized import parameterized
from time import perf_counter

from games import (
    MIX_MNE,
    SECURITY_MNE,
    MIX_STRATEGIES,
    FINITE_MIX_STRATEGIES,
    PURE_STRATEGIES,
    report_solutions,
    create_report_preprocessed,
    UncertaintyParams,
    logger
)
from games.access import get_game_factorization
from games.solve.solution import solve1
from games.reports_performance import report_performance
from games.performance import get_initialized_game_performance, GamePerformance

from possibilities import PossibilitySet, PossibilityDist
from preferences.preferences_probability import ProbPrefExpectedValue
from preferences import SetPreference1

from factorization.structures import FactorizationSolverParams, FactorizationSolverSpec
from factorization.solve_n_players import preprocess_n_player_game
from factorization.algos_factorization import get_game_factorization_no_collision_check

from toy_games.n_player_toy_game import get_toy_car_game
from toy_games.n_player_toy_game_zoo import (
    toy_params_x,
    toy_params_star,
    toy_params_x_with_base,
    toy_params_indep_lanes,
    toy_params_one_indep_lane,
    toy_params_two_indep_games,
    toy_params_two_x_joint,
    toy_params_two_x_crossed
)

uncertainty_sets = UncertaintyParams(poss_monad=PossibilitySet(), mpref_builder=SetPreference1)
uncertainty_prob = UncertaintyParams(poss_monad=PossibilityDist(), mpref_builder=ProbPrefExpectedValue)


uncertainty_params = [
    [uncertainty_sets, "sets"],
    [uncertainty_prob, "prob"],
]

toy_game_params = [
    toy_params_x,
    toy_params_star,
    toy_params_x_with_base,
    toy_params_indep_lanes,
    toy_params_one_indep_lane,
    toy_params_two_indep_games,
    toy_params_two_x_joint,
    toy_params_two_x_crossed,
]

strategies = [
    PURE_STRATEGIES,
    # MIX_STRATEGIES
    FINITE_MIX_STRATEGIES
]

nash_strategy = [
    MIX_MNE,
    # SECURITY_MNE
]

use_factorization = [
    # [True, get_game_factorization, "base"],
    [True, get_game_factorization_no_collision_check, "no_col"],
    [False, None]
]

params = list(product(toy_game_params, uncertainty_params, strategies, nash_strategy, use_factorization))


@parameterized(params)
def test_n_player_toy_game(toy_game_parameters, uncert_params, strat, nash_strat, use_fact):
    """
    N-Player toy game
    """
    runs = 1
    r_run = 0

    d = "out/"
    game_name = f"{toy_game_parameters.params_name}"
    solver_name = f"{strat}-{nash_strat}-{uncert_params[1]}{'-fact_' + use_fact[2] if use_fact[0] else ''}"
    logger.info(f"Start test: {game_name} with solver params {solver_name}")
    game = get_toy_car_game(toy_games_params=toy_game_parameters, uncertainty_params=uncert_params[0])

    get_factorization = use_fact[1]  # factorization algo used
    dt = toy_game_parameters.dt  # delta-t of discretization
    admissible_strategies = strat
    strategy_multiple_nash = nash_strat

    solve_params = FactorizationSolverParams(
            admissible_strategies=admissible_strategies,
            strategy_multiple_nash=strategy_multiple_nash,
            dt=dt,
            use_factorization=use_fact[0],
            get_factorization=get_factorization
    )
    solver_spec = FactorizationSolverSpec(solver_name, solve_params)

    dg = join(d, game_name)
    ds = join(dg, solver_name)

    list_game_perf = []
    for i in range(runs):
        game_performance: GamePerformance = get_initialized_game_performance(game=game, solver_params=solve_params)

        # start performance counter collect time used for preprocessing
        t1 = perf_counter()

        game_preprocessed = preprocess_n_player_game(game, solver_spec.solver_params, game_perf=game_performance)

        # stop counter and collect performance
        t2 = perf_counter()
        game_performance.pre_pro_player_pi.total_time = t2 - t1

        solutions = solve1(game_preprocessed, game_perf=game_performance)

        logger.info("Game Performance", game_performance=game_performance)

        list_game_perf.append(game_performance)
        if r_run == i:

            r_solutions = report_solutions(game_preprocessed, solutions)
            r_preprocessed = create_report_preprocessed(game_name, game_preprocessed)

            r_solutions.to_html(join(ds, "r_solutions.html"))
            r_preprocessed.to_html(join(ds, "r_preprocessed.html"))

    r_performance = report_performance(list_game_perf=list_game_perf)
    r_performance.to_html(join(ds, "r_performance.html"))
