from os.path import join
from itertools import product
from time import perf_counter

from parameterized import parameterized

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


from factorization.structures import FactorizationSolverParams, FactorizationSolverSpec
from factorization.algos_factorization import get_game_factorization_no_collision_check
from factorization.solve_n_players import preprocess_n_player_game

from duckie_games.game_generation import get_duckie_game

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
    # [uncertainty_sets, "sets"],
    [uncertainty_prob, "prob"],
]

duckie_game_params = [
    # two_player_4way,
    # two_player_4way_intersection_only,
    # three_player_4way,
    three_player_4way_intersection_only,
    # three_player_4way_double,
    # three_player_4way_double_intersection_only,
]

strategies = [
    PURE_STRATEGIES,
    # MIX_STRATEGIES
    # FINITE_MIX_STRATEGIES
]

nash_strategy = [
    MIX_MNE,
    # SECURITY_MNE
]

use_factorization = [
    # [True, get_game_factorization, "base"],
    [True, get_game_factorization_no_collision_check, "no_col"],
    # [False, None]
]

params = list(product(duckie_game_params, uncertainty_params, strategies, nash_strategy, use_factorization))


@parameterized(params)
def test_duckie_games(duckie_game_parameters, duckie_uncert_params, strat, nash_strat, use_fact):
    """
    n-player duckie game tests
    """
    runs = 1
    r_run = 0

    logger.info(f"Starting test: {duckie_game_parameters.desc}")
    d = "out/"
    game_name = _get_game_name(duckie_game_parameters)

    solver_name = f"{strat}-{nash_strat}-{duckie_uncert_params[1]}{'-fact_' + use_fact[2] if use_fact[0] else ''}"
    game = get_duckie_game(duckie_game_params=duckie_game_parameters, uncertainty_params=duckie_uncert_params[0])
    use_factorization = use_fact[0]
    get_factorization = use_fact[1]  # factorization algo used
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
            # print(solutions.game_solution.policies)
            # print(solutions.game_solution.states_to_solution)

    r_performance = report_performance(list_game_perf=list_game_perf)
    r_performance.to_html(join(ds, "r_performance.html"))


def _get_game_name(duckie_game_parameters):
    player_number = duckie_game_parameters.player_number
    m = duckie_game_parameters.map_name
    accels = duckie_game_parameters.available_accels
    game_name = f"{player_number}_player_{m}"
    for ac in accels.values():
        game_name += f"_{len(ac)}"
    return game_name
