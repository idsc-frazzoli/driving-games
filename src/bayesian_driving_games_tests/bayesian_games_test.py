import unittest
from decimal import Decimal as D
from os.path import join
from bayesian_driving_games.game_generation import get_bayesian_driving_game
from driving_games import (
    TwoVehicleSimpleParams,
    NO_LIGHTS,
    UncertaintyParams,
    ProbPrefExpectedValue,
    PossibilityDist,
)
from games import (
    MIX_MNE,
    SolverParams,
    report_solutions,
    create_report_preprocessed,
)
from games_zoo.solvers import SolverSpec
from bayesian_driving_games.solution import solve_bayesian_game
from bayesian_driving_games.preprocess import preprocess_bayesian_game


@unittest.skip("Bayesian games to be refactored in the future")
def test2():
    # todo readjust
    road = D(5)
    side = D(4)
    p0 = TwoVehicleSimpleParams(
        side=D(7),
        road=road,
        road_lane_offset=road / 2,  # center
        max_speed=D(5),
        min_speed=D(0.1),
        max_wait=D(1),
        available_accels=frozenset({D(-2), D(-1), D(0), D(+1)}),
        collision_threshold=3.0,
        light_actions=frozenset({NO_LIGHTS}),
        dt=D(1),
        first_progress=D(0),
        second_progress=D(0),
        shared_resources_ds=D(1.5),
    )
    # uncertainty_sets = TwoVehicleUncertaintyParams(poss_monad=PossibilitySet(), mpref_builder=SetPreference1)
    uncertainty_prob = UncertaintyParams(poss_monad=PossibilityDist(), mpref_builder=ProbPrefExpectedValue)
    d = "out/bayesian_dg/"
    game2 = get_bayesian_driving_game(p0, uncertainty_prob)
    game_name = "50-50,test1 : aggressive"
    solver_spec = SolverSpec("test", SolverParams(D(1), MIX_MNE, False))
    solver_name = solver_spec.desc
    game_preprocessed = preprocess_bayesian_game(game2, solver_spec.solver_params)
    solutions = solve_bayesian_game(game_preprocessed)
    dg = join(d, game_name)
    ds = join(dg, solver_name)
    r_solutions = report_solutions(game_preprocessed, solutions)
    r_preprocessed = create_report_preprocessed(game_name, game_preprocessed)

    r_solutions.to_html(join(ds, "r_solutions.html"))
    r_preprocessed.to_html(join(ds, "r_preprocessed.html"))
    # print(solutions.game_solution.policies)
    # print(solutions.game_solution.states_to_solution)
