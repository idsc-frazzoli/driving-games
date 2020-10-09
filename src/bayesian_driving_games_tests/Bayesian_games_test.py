from decimal import Decimal as D
from os.path import join

from bayesian_driving_games.game_generation import get_bayesian_driving_game
from bayesian_driving_games.solution import solve_bayesian_game
from belief_games import TwoVehicleSimpleParams, NO_LIGHTS, TwoVehicleUncertaintyParams, PossibilitySet, SetPreference1, \
    ProbPrefExpectedValue, ProbabilityFraction, get_two_vehicle_game
from games import STRATEGY_MIX, SolverParams, preprocess_game, solve1, report_solutions, create_report_preprocessed
from games_scripts.solvers import SolverSpec


def test2():
    road = D(6)
    side = D(8)
    p0 = TwoVehicleSimpleParams(
        side=D(8),
        road=road,
        road_lane_offset=road / 2,  # center
        max_speed=D(5),
        min_speed=D(1),
        max_wait=D(1),
        available_accels=frozenset({D(-2), D(-1), D(0), D(+1)}),
        collision_threshold=3.0,
        light_actions=frozenset({NO_LIGHTS}),
        dt=D(1),
        first_progress=D(0),
        second_progress=D(0),
        shared_resources_ds=D(1.5),
    )
    uncertainty_sets = TwoVehicleUncertaintyParams(poss_monad=PossibilitySet(), mpref_builder=SetPreference1)
    uncertainty_prob = TwoVehicleUncertaintyParams(
        poss_monad=ProbabilityFraction(), mpref_builder=ProbPrefExpectedValue)
    d = "ml_out/tests/"
    game2 = get_bayesian_driving_game(p0,uncertainty_sets)
    # game1 = get_master_slave_game(p0,uncertainty_sets,2)
    game_name = "test_ml"
    solver_spec = SolverSpec("test", SolverParams(D(1), STRATEGY_MIX, False))
    solver_name = solver_spec.desc
    game_preprocessed = preprocess_game(game2, solver_spec.solver_params)
    solutions = solve_bayesian_game(game_preprocessed)
    dg = join(d, game_name)
    ds = join(dg, solver_name)
    r_solutions = report_solutions(game_preprocessed, solutions)
    r_preprocessed = create_report_preprocessed(game_name, game_preprocessed)

    r_solutions.to_html(join(ds, "r_solutions.html"))
    r_preprocessed.to_html(join(ds, "r_preprocessed.html"))
    print("end!")
