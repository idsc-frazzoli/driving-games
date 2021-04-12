from itertools import product
import math

from parameterized import parameterized

from games import (
    MIX_MNE,
    SECURITY_MNE,
    MIX_STRATEGIES,
    FINITE_MIX_STRATEGIES,
    PURE_STRATEGIES,
    logger
)

from games.access import get_game_factorization

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
    three_player_4way,
    three_player_4way_intersection_only,
    three_player_4way_double,
    three_player_4way_double_intersection_only,
    uncertainty_prob,
    uncertainty_sets,
)


uncertainty_params = [
    [uncertainty_prob, "prob"],
    [uncertainty_sets, "sets"]
]

duckie_game_params = [
    two_player_4way_intersection_only,
    # three_player_4way_intersection_only,
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


betas = [
    0,  # resources of game
    math.inf  # forward reachable set
]

params = list(product(duckie_game_params, uncertainty_params, strategies, nash_strategy, betas))


@parameterized(params)
def test_duckie_games(duckie_game_parameters, duckie_uncert_params, strat, nash_strat, beta):
    """
    This functions compares the partitioning of players for the different factorization algorithms with the base
    factorization algorithm. Only for a two player game!
    """

    game = get_duckie_game(duckie_game_params=duckie_game_parameters, uncertainty_params=duckie_uncert_params[0])
    dt = duckie_game_parameters.dt  # delta-t of discretization
    admissible_strategies = strat
    strategy_multiple_nash = nash_strat

    if not (beta != 0 and len(game.players) > 2):
        solve_params_base = FactorizationSolverParams(
            admissible_strategies=admissible_strategies,
            strategy_multiple_nash=strategy_multiple_nash,
            dt=dt,
            use_factorization=True,
            get_factorization=get_game_factorization,
            beta=beta
        )
        solver_spec = FactorizationSolverSpec("Fact Test", solve_params_base)

        game_preprocessed_base = preprocess_n_player_game(game, solver_spec.solver_params)

        factorization_algo = [
            [get_game_factorization_no_collision_check, "no_col"],
            [get_game_factorization_as_create_game_graph, "as_gg"],
            [get_game_factorization_n_players_as_create_game_graph, "n_player_as_gg"]
        ]

        for algo, name in factorization_algo:
            logger.info(f"Testing {name} against base")
            solve_params_test = FactorizationSolverParams(
                admissible_strategies=admissible_strategies,
                strategy_multiple_nash=strategy_multiple_nash,
                dt=dt,
                use_factorization=True,
                get_factorization=algo,
                beta=beta
            )
            solver_spec_test = FactorizationSolverSpec(name, solve_params_test)

            game_preprocessed_test = preprocess_n_player_game(game, solver_spec_test.solver_params)

            for js in game_preprocessed_test.game_factorization.ipartitions:
                partition_base = game_preprocessed_base.game_factorization.ipartitions[js]
                partition_test = game_preprocessed_test.game_factorization.ipartitions[js]
                assert partition_base == partition_test, (  # check the partitions found in the newer algo with the base
                    f"Partition test {partition_test} is not {partition_base} for state \n"
                    f"{js}"
                )
