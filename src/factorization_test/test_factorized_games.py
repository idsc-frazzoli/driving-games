from decimal import Decimal as D
from os.path import join

from games import (
    MIX_MNE,
    PURE_STRATEGIES,
    report_solutions,
    create_report_preprocessed,
)
from games.access import get_game_factorization
from games.solve.solution import solve1

from factorization.structures import FactorizationSolverParams, FactorizationSolverSpec
from factorization.solve import preprocess_game_factorization
from factorization.game_generation import get_duckie_game


def test_factorization():
    d = "out/factorized_dg/"
    game_name = "factorized game"
    solver_name = "Test"
    game_spec =  get_duckie_game()  # todo generate a duckie game
    game = game_spec.game
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
            get_factorization=get_factorization)
    solver_spec = FactorizationSolverSpec("test", solve_params)

    game_preprocessed = preprocess_game_factorization(game, solver_spec.solver_params)
    solutions = solve1(game_preprocessed)

    dg = join(d, game_name)
    ds = join(dg, solver_name)
    r_solutions = report_solutions(game_preprocessed, solutions)
    r_preprocessed = create_report_preprocessed(game_name, game_preprocessed)

    r_solutions.to_html(join(ds, "r_solutions.html"))
    r_preprocessed.to_html(join(ds, "r_preprocessed.html"))
    # print(solutions.game_solution.policies)
    # print(solutions.game_solution.states_to_solution)
