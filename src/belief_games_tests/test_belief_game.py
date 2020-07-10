from decimal import Decimal as D
from os.path import join

from belief_games import get_leader_follower_game, get_asym, get_asym_lf, TwoVehicleSeenObservation
from belief_games.belief_calculation import belief_calc_simple1, belief_calc_simple2
from driving_games import get_alone, NO_LIGHTS, get_two_vehicle_game, TwoVehicleSimpleParams
from games import preprocess_game, solve1, SolverParams, STRATEGY_SECURITY, STRATEGY_MIX, report_solutions, \
    create_report_preprocessed
from games.get_indiv_games import get_individual_games
from games_scripts.solvers import SolverSpec


def test1():
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
            solutions = solve1(game_seen_preprocessed)

        else:
            game1 = get_leader_follower_game(p0)
            game2 = get_leader_follower_game(p0)
            game3 = get_leader_follower_game(p0)
            # game3 = get_individual_games(game1)
            game4 = get_leader_follower_game(p0)

            solver_spec = SolverSpec("test", SolverParams(D(1), STRATEGY_MIX, False))
            solver_name = solver_spec.desc

            game1_preprocessed = preprocess_game(game1, solver_spec.solver_params)
            solutions1 = solve1(game1_preprocessed)

            # game1_preprocessed = preprocess_game(game2, solver_spec.solver_params)
            # solutions1 = solve1(game1_preprocessed)
            # game3_preprocessed = preprocess_game(game3, solver_spec.solver_params)
            # solutions3 = solve1(game3_preprocessed)
            # game4_preprocessed = preprocess_game(game3, solver_spec.solver_params)
            # solutions4 = solve1(game4_preprocessed)

            r1, r2, r3, r4 = [], [], [], []

            for joint_state, node in solutions1.game_solution.states_to_solution.items():
                print(r1)
                print(list(solutions1.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0])
                if list(solutions1.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0].joint.active == False:
                    r1.append(list(solutions1.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0].personal.duration)
                else:
                    pass
                # if list(solutions2.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0].joint.active == False:
                #     r2 = r2.append(list(solutions2.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0].personal.duration)
                # else:
                #     pass
                # if list(solutions3.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0].joint.active == False:
                #     r3 = r3.append(list(solutions3.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0].personal.duration)
                # else:
                #     pass
                # if list(solutions4.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0].joint.active == False:
                #     r4 = r4.append(list(solutions4.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0].personal.duration)
                # else:
                #     pass


            # TODO Values of the game.


            probab1 = belief_calc_simple1(p0)
            probab2 = belief_calc_simple2(p0)

            weighted_r11 = probab1["Game1"] * r1
            weighted_r12 = probab1["Game2"] * r2
            weighted_r13 = probab1["Game3"] * r3
            weighted_r14 = probab1["Game4"] * r4
            weighted_r1 = [weighted_r11, weighted_r12, weighted_r13, weighted_r14]

            weighted_r21 = probab1["Game1"] * r1
            weighted_r22 = probab1["Game2"] * r2
            weighted_r23 = probab1["Game3"] * r3
            weighted_r24 = probab1["Game4"] * r4
            weighted_r2 = [weighted_r21, weighted_r22, weighted_r23, weighted_r24]

            # TODO Update step, solution states are needed.
            if max(weighted_r1) == weighted_r11:
                p0.first_progress = p0.first_progress + 1
            elif max(weighted_r1) == weighted_r12:
                p0.first_progress = p0.first_progress + 1
            elif max(weighted_r1) == weighted_r13:
                p0.first_progress = p0.first_progress + 1
            elif max(weighted_r1) == weighted_r14:
                p0.first_progress = p0.first_progress + 1
            else:
                print("Something went wrong in the state update of car 1 of the belief game algorithm")

            if max(weighted_r2) == weighted_r21:
                p0.second_progress = p0.second_progress + 2
            elif max(weighted_r2) == weighted_r22:
                p0.second_progress = p0.second_progress + 2
            elif max(weighted_r2) == weighted_r23:
                p0.second_progress = p0.second_progress + 2
            elif max(weighted_r2) == weighted_r24:
                p0.second_progress = p0.second_progress + 2
            else:
                print("Something went wrong in the state update of car 2 of the belief game algorithm")

            seen = obs.do_we_see_us(p0.first_progress, p0.second_progress)





    # d = "ml_out/tests/"
    # game_spec = get_asym_lf()
    # game1 = game_spec.game
    # game_name = "test_ml"
    # solver_spec = SolverSpec("test", SolverParams(D(1), STRATEGY_MIX, False))
    # solver_name = solver_spec.desc
    # game_preprocessed = preprocess_game(game1, solver_spec.solver_params)
    # solutions = solve1(game_preprocessed)
    # for state, solution in solutions.game_solution.states_to_solution.items():
    #     print("Joint state:\n", state)
    #     print("Values and actions:\n", solution.solved)
    #     print("Game values:\n", solution.va.game_value)
    # d = "ml_out/tests/"
    # dg = join(d, game_name)
    # ds = join(dg, solver_name)
    # r_solutions = report_solutions(game_preprocessed, solutions)
    # r_preprocessed = create_report_preprocessed(game_name, game_preprocessed)
    #
    # r_solutions.to_html(join(ds, "r_solutions.html"))
    # r_preprocessed.to_html(join(ds, "r_preprocessed.html"))

    print("end!!")
