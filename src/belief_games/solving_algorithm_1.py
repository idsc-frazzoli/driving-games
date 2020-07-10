import numpy as np
from decimal import Decimal as D

from belief_games import TwoVehicleSimpleParams, TwoVehicleSeenObservation, get_two_vehicle_game, NO_LIGHTS, \
    get_leader_follower_game
from belief_games.belief_calculation import belief_calc_simple1, belief_calc_simple2
from games import SolverParams, STRATEGY_MIX, preprocess_game, solve1
from games.get_indiv_games import get_individual_games
from games_scripts.solvers import SolverSpec


def solving_belief_games():
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

    start = p0.side + p0.road_lane_offset
    L = p0.side + p0.road + p0.side

    p1_ref = (D(start), D(0), D(+90))
    p2_ref = (D(L), D(start), D(-180))

    obs = TwoVehicleSeenObservation(road, side, p1_ref, p2_ref)
    seen = obs.do_we_see_us(p0.first_progress, p0.second_progress)

    while True:

        if seen:
            game_seen = get_two_vehicle_game(p0)
            solver_spec = SolverSpec("test", SolverParams(D(1), STRATEGY_MIX, False))
            solver_name = solver_spec.desc
            game_seen_preprocessed = preprocess_game(game_seen, solver_spec.solver_params)
            solutions1 = solve1(game_seen_preprocessed)

        else:
            game1 = get_leader_follower_game(p0)
            game2 = get_two_vehicle_game(p0)
            game3 = get_individual_games(game1)
            game4 = get_leader_follower_game(p0)

            solver_spec = SolverSpec("test", SolverParams(D(1), STRATEGY_MIX, False))
            solver_name = solver_spec.desc

            game1_preprocessed = preprocess_game(game1, solver_spec.solver_params)
            solutions1 = solve1(game1_preprocessed)
            game2_preprocessed = preprocess_game(game2, solver_spec.solver_params)
            solutions2 = solve1(game2_preprocessed)
            game3_preprocessed = preprocess_game(game3, solver_spec.solver_params)
            solutions3 = solve1(game3_preprocessed)

            r1 = 1
            r2 = 2
            r3 = 3
            r4 = 4

            probab1 = belief_calc_simple1(p0)
            probab2 = belief_calc_simple2(p0)

            weighted_r1 = probab1["Game1"] * r1
            weighted_r2 = probab1["Game2"] * r2
            weighted_r3 = probab1["Game3"] * r3
            weighted_r4 = probab1["Game4"] * r4
            weighted_r = [weighted_r1, weighted_r2, weighted_r3, weighted_r4]

            if max(weighted_r) == weighted_r1:
                p0.first_progress = p0.first_progress + 1
                p0.second_progress = p0.second_progress + 2
            elif max(weighted_r) == weighted_r2:
                p0.first_progress = p0.first_progress + 1
                p0.second_progress = p0.second_progress + 2
            elif max(weighted_r) == weighted_r2:
                p0.first_progress = p0.first_progress + 1
                p0.second_progress = p0.second_progress + 2
            elif max(weighted_r) == weighted_r2:
                p0.first_progress = p0.first_progress + 1
                p0.second_progress = p0.second_progress + 2
            else:
                print("Something went wrong in the state update of the belief game algorithm")

            seen = obs.do_we_see_us(p0.first_progress, p0.second_progress)

    # d = "ml_out/tests/"
    # game_spec = get_asym_lf()
    # game1 = game_spec.game
    # game_name = "test_ml"
    # solver_spec = SolverSpec("test", SolverParams(D(1), STRATEGY_MIX, False))
    # solver_name = solver_spec.desc
    # game_preprocessed = preprocess_game(game1, solver_spec.solver_params)
    # solutions = solve1(game_preprocessed)
