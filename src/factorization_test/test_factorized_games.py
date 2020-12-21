from decimal import Decimal as D
from os.path import join
from driving_games import (
    TwoVehicleSimpleParams,
    NO_LIGHTS,
    TwoVehicleUncertaintyParams,
    ProbPrefExpectedValue,
    ProbabilityFraction,
)
from games import (
    STRATEGY_MIX,
    SolverParams,
    report_solutions,
    create_report_preprocessed,
)
from games_zoo.solvers import SolverSpec



def test():
    d = "out/factorized_dg/"
    game = # todo get_n_player_game()
    game_name = "factorized game"
    solver_spec = SolverSpec("test", SolverParams(D(1), STRATEGY_MIX, False))
    solver_name = solver_spec.desc
    game_preprocessed = # todo preprocessed_game_factorized
    solutions = # todo solve_factorized_game
    dg = join(d, game_name)
    ds = join(dg, solver_name)
    r_solutions = report_solutions(game_preprocessed, solutions)
    r_preprocessed = create_report_preprocessed(game_name, game_preprocessed)

    r_solutions.to_html(join(ds, "r_solutions.html"))
    r_preprocessed.to_html(join(ds, "r_preprocessed.html"))
    # print(solutions.game_solution.policies)
    # print(solutions.game_solution.states_to_solution)
