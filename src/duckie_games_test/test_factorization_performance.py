from os.path import join
from itertools import product
from time import perf_counter
import math
from dataclasses import replace
from toolz import valmap
from decimal import Decimal as D

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

from games.game_def import Game
from games.access import get_game_factorization
from games.solve.solution import solve1
from games.reports_performance import report_performance
from games.performance import get_initialized_game_performance, GamePerformance
from games.utils import fvalmap

from factorization.structures import FactorizationSolverParams, FactorizationSolverSpec
from factorization.algos_factorization import (
    get_game_factorization_no_collision_check,
    get_game_factorization_as_create_game_graph,
    get_game_factorization_n_players_as_create_game_graph,
)
from factorization.solve_n_players import preprocess_n_player_game

from duckie_games.game_generation import get_duckie_game

from duckie_games.zoo import (
    two_player_4way,
    two_player_4way_intersection_only,
    two_player_roundabout_only,
    three_player_4way,
    three_player_4way_intersection_only,
    three_player_4way_double,
    three_player_4way_double_intersection_only,
    uncertainty_prob,
    uncertainty_sets,
)


uncertainty_params = [
    [uncertainty_sets, "sets"],
    [uncertainty_prob, "prob"],
]

duckie_game_params = [
    # two_player_4way,
    # two_player_4way_intersection_only,
    two_player_roundabout_only
    # three_player_4way,
    # three_player_4way_intersection_only,
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
    # [True, get_game_factorization_no_collision_check, "no_col"],
    # [True, get_game_factorization_as_create_game_graph, "as_gg"],
    [True, get_game_factorization_n_players_as_create_game_graph, "n_play_as_gg"],
    [False, None]
]

betas = [
    0, # resources of game
    # 0.2,
    # 1,
    # 5,
    math.inf  # forward reachable set
]

# during the test the game is solved for each set of accelerations specified, we differentiate between the set
# monad and the probability monad
accelerations_test ={
    "prob" : [
      # ["0", "+0.5"],
      # ["-1", "0", "+0.5"],
      # ["-1", "0", "+0.5", "+1"],
      # ["-1", "0", "+0.5", "+1", "+2"],
        ["-1", "+0", "+1", "+1.5"]
      ],

    "sets" : [
      # ["0", "+0.5"],
      # ["-1", "0", "+0.5"],
      # ["-1", "0", "+0.5", "+1"],
      # ["-2", "-1", "0", "+0.5", "+1"],
      # ["-2", "-1", "-0.5", "0", "+0.5", "+1"],
      # ["-2", "-1", "-0.5", "0", "+0.5", "+1", "+1.5"],
      # ["-2", "-1.5", "-1", "-0.5", "0", "+0.5", "1", "+1.5"],
        ["-1", "+0", "+1", "+1.5"]
    ]
}

params = list(product(duckie_game_params, uncertainty_params, strategies, nash_strategy, use_factorization, betas))

@parameterized(params)
def test_factorization(duckie_game_parameters, duckie_uncert_params, strat, nash_strat, use_fact, beta):
    """
    Test the factorization algos for different size of action sets
    """

    if beta is not math.inf and not use_fact[0]:
        # When no factorization is used, only run it for beta=inf
        return

    upsample_vis = 2  # divides the timestep used for upsampling in simulation

    runs = 1  # how many times should the game be solved (for the performance info)
    # fixme for more than 1 run the results for the runs > 1 are different because of caching some functions
    r_run = 0  # at which run should the report of the simulation be created

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
            get_factorization=get_factorization,
            beta=beta
    )
    solver_name = (
        f"{strat}-{nash_strat}-{duckie_uncert_params[1]}"
        f"{'-fact_' + use_fact[2] + '_beta=' + str(beta) if use_fact[0] else ''}"
    )
    solver_spec = FactorizationSolverSpec(solver_name, solve_params)

    logger.info(f"Starting test: {duckie_game_parameters.desc}")
    d = "out/tests/fact_perf_test"  # out tests in order to collect reports in circleci
    game_name = _get_game_name_without_accel(duckie_game_parameters)
    dg = join(d, game_name)
    ds = join(dg, solver_name)

    logger.info(game_name=game_name, solver_name=solver_name)

    list_game_perf = []  # initialize empty list for the game performances of each run

    # take either the set for the probability monad or the set monad
    available_acels = accelerations_test[duckie_uncert_params[1]]
    for accel_set in available_acels:  # iterate through all the sets of accelerations

        accel_set = list(map(D, accel_set))  # convert to decimals

        # replace the accelerations defined in the zoo module by the new set of accelerations
        old_accelerations = duckie_game_parameters.available_accels
        new_accelerations = valmap(lambda _: accel_set, old_accelerations)
        new_duckie_parameters = replace(duckie_game_parameters, available_accels=new_accelerations, lanes=None)

        ds_reports = join(ds, _get_number_of_actions(new_duckie_parameters))

        game = get_duckie_game(
            duckie_game_params=new_duckie_parameters,
            uncertainty_params=duckie_uncert_params[0]
        )

        for i in range(runs):
            # initialize an empty game performance info object
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
                r_solutions = report_solutions(game_preprocessed, solutions, upsample_log=upsample_vis)
                r_preprocessed = create_report_preprocessed(game_name, game_preprocessed)

                r_solutions.to_html(join(ds_reports, "r_solutions.html"))
                r_preprocessed.to_html(join(ds_reports, "r_preprocessed.html"))
                # print(solutions.game_solution.policies)
                # print(solutions.game_solution.states_to_solution)

        r_performance = report_performance(list_game_perf=list_game_perf)
        r_performance.to_html(join(ds, "r_performance.html"))


def _get_game_name_without_accel(duckie_game_parameters):
    """
    Returns the folder name without the number accelerations per player
    """
    player_number = duckie_game_parameters.player_number
    m = duckie_game_parameters.map_name
    game_name = f"{player_number}_player_{m}"
    return game_name


def _get_number_of_actions(duckie_game_parameters):
    """
    Returns the number of accelerations per player as string
    """
    accels = duckie_game_parameters.available_accels
    nb_accels = f""
    for i, ac in enumerate(accels.values()):
        if i == 0:
            nb_accels += f"{len(ac)}"
        else:
            nb_accels += f"_{len(ac)}"
    return nb_accels

