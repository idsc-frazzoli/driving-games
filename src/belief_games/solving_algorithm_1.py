from decimal import Decimal as D, Decimal

from belief_games import get_master_slave_game, get_asym, TwoVehicleSeenObservation
from belief_games.belief_calculation import belief_calc_simple1, belief_calc_simple2
from driving_games import (
    NO_LIGHTS,
    get_two_vehicle_game,
    TwoVehicleSimpleParams,
    PlayerName,
    TwoVehicleUncertaintyParams,
    PossibilitySet,
    SetPreference1,
    frozendict,
    VehicleState,
)
from games import preprocess_game, solve1, SolverParams, STRATEGY_SECURITY, STRATEGY_MIX
from games_scripts.solvers import SolverSpec


def solving_hidden_agent_game():

    # Define Parameters
    road = D(6)
    side = D(8)
    # r1, r2, = [], []
    # prob1, prob2 = [], []

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

    # starting positons of the two cars
    start = p0.side + p0.road_lane_offset
    L = p0.side + p0.road + p0.side
    initial_state1 = {
        "N↑": VehicleState(
            ref=(Decimal("11"), Decimal("0"), Decimal("90")),
            x=Decimal("0"),
            v=Decimal("1"),
            wait=Decimal("0"),
            light="none",
        ),
        "W←": VehicleState(
            ref=(Decimal("22"), Decimal("11"), Decimal("-180")),
            x=Decimal("0"),
            v=Decimal("1"),
            wait=Decimal("0"),
            light="none",
        ),
    }
    initial_state1 = frozendict(initial_state1)
    initial_state2 = {
        "level-1": VehicleState(
            ref=(Decimal("11"), Decimal("0"), Decimal("90")),
            x=Decimal("0"),
            v=Decimal("1"),
            wait=Decimal("0"),
            light="none",
        ),
        "level-0": VehicleState(
            ref=(Decimal("22"), Decimal("11"), Decimal("-180")),
            x=Decimal("0"),
            v=Decimal("1"),
            wait=Decimal("0"),
            light="none",
        ),
    }
    initial_state2 = frozendict(initial_state2)
    initial_state3 = {
        "level-0": VehicleState(
            ref=(Decimal("11"), Decimal("0"), Decimal("90")),
            x=Decimal("0"),
            v=Decimal("1"),
            wait=Decimal("0"),
            light="none",
        ),
        "level-1": VehicleState(
            ref=(Decimal("22"), Decimal("11"), Decimal("-180")),
            x=Decimal("0"),
            v=Decimal("1"),
            wait=Decimal("0"),
            light="none",
        ),
    }
    initial_state3 = frozendict(initial_state3)

    p1_ref = (D(start), D(0), D(+90))
    p2_ref = (D(L), D(start), D(-180))

    # See if the two cars initially see each other
    obs = TwoVehicleSeenObservation(road, side, p1_ref, p2_ref)
    seen = obs.do_we_see_us(p0.first_progress, p0.second_progress)

    # Loop starts here
    while True:

        # Every timestep, determine whether the cars see each other. If yes, play a normal intersection game and finish.
        if seen:
            game_seen = get_two_vehicle_game(p0, uncertainty_sets)
            solver_spec = SolverSpec("test", SolverParams(D(1), STRATEGY_MIX, False))
            solver_name = solver_spec.desc
            game_seen_preprocessed = preprocess_game(game_seen, solver_spec.solver_params)
            solutions = solve1(game_seen_preprocessed)
            print("normal game starting at: ", p0.first_progress, "and ", p0.second_progress)
            break

        # If no, do the following
        else:

            # Define the four games
            game2 = get_master_slave_game(p0, uncertainty_sets, 2)
            game3 = get_master_slave_game(p0, uncertainty_sets, 1)
            game4 = get_two_vehicle_game(p0, uncertainty_sets)

            solver_spec = SolverSpec("test", SolverParams(D(1), STRATEGY_MIX, False))
            solver_name = solver_spec.desc

            # Preprocess and solve games. The alone game ("Game1") gives an error at the moment, because when
            # creating the game graph, the number of players is hardcoded to 2 at the moment...

            game3_preprocessed = preprocess_game(game3, solver_spec.solver_params)
            solutions3 = solve1(game3_preprocessed)
            game4_preprocessed = preprocess_game(game4, solver_spec.solver_params)
            solutions4 = solve1(game4_preprocessed)
            game2_preprocessed = preprocess_game(game2, solver_spec.solver_params)
            solutions2 = solve1(game2_preprocessed)

            #
            # #######
            a4_P1 = list(
                solutions4.game_solution.states_to_solution[initial_state1].va.game_value["N↑"].support()
            )
            res = 0
            n = 0
            for i in range(len(a4_P1)):
                if a4_P1[i].joint == None:
                    res = res + a4_P1[i].personal.duration
                    n = n + 1
            game_value4_P1 = res / n

            a4_P2 = list(
                solutions4.game_solution.states_to_solution[initial_state1].va.game_value["W←"].support()
            )
            res = 0
            n = 0
            for i in range(len(a4_P2)):
                if a4_P2[i].joint == None:
                    res = res + a4_P2[i].personal.duration
                    n = n + 1
            game_value4_P2 = res / n

            a3_P1 = list(
                solutions3.game_solution.states_to_solution[initial_state3].va.game_value["level-0"].support()
            )
            res = 0
            n = 0
            for i in range(len(a3_P1)):
                if a3_P1[i].joint == None:
                    res = res + a3_P1[i].personal.duration
                    n = n + 1
            game_value3_P1 = res / n

            a3_P2 = list(
                solutions3.game_solution.states_to_solution[initial_state3].va.game_value["level-1"].support()
            )
            res = 0
            n = 0
            for i in range(len(a3_P2)):
                if a3_P2[i].joint == None:
                    res = res + a3_P2[i].personal.duration
                    n = n + 1
            game_value3_P2 = res / n

            a2_P1 = list(
                solutions2.game_solution.states_to_solution[initial_state2].va.game_value["level-1"].support()
            )
            res = 0
            n = 0
            for i in range(len(a2_P1)):
                if a2_P1[i].joint == None:
                    res = res + a2_P1[i].personal.duration
                    n = n + 1
            game_value2_P1 = res / n

            a2_P2 = list(
                solutions2.game_solution.states_to_solution[initial_state2].va.game_value["level-0"].support()
            )
            res = 0
            n = 0
            for i in range(len(a2_P2)):
                if a2_P2[i].joint == None:
                    res = res + a2_P2[i].personal.duration
                    n = n + 1
            game_value2_P2 = res / n

            # My try of getting a hand on any values of the game => must be changed.
            # for joint_state, node in solutions1.game_solution.states_to_solution.items():
            #     print(list(solutions1.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0])
            #     if list(solutions1.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0].joint.active == False:
            #         r1.append(list(solutions1.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0].personal.duration)
            #     else:
            #         pass
            #     if list(solutions2.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0].joint.active == False:
            #         r2 = r2.append(list(solutions2.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0].personal.duration)
            #     else:
            #         pass
            #     if list(solutions3.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0].joint.active == False:
            #         r3 = r3.append(list(solutions3.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0].personal.duration)
            #     else:
            #         pass
            #     if list(solutions4.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0].joint.active == False:
            #         r4 = r4.append(list(solutions4.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0].personal.duration)
            #     else:
            #         pass
            ######

            # calculate the probabilities of playing each game.
            probab1 = belief_calc_simple1(p0)
            probab2 = belief_calc_simple2(p0)

            expected_game_value2_P1 = probab1["Game_SM"] * game_value2_P1
            expected_game_value2_P2 = probab2["Game_SM"] * game_value2_P2
            expected_game_value3_P1 = probab1["Game_MS"] * game_value3_P1
            expected_game_value3_P2 = probab2["Game_MS"] * game_value3_P2
            expected_game_value4_P1 = probab1["Game_Normal"] * game_value4_P1
            expected_game_value4_P2 = probab2["Game_Normal"] * game_value4_P1

            l1 = min(expected_game_value2_P1, expected_game_value3_P1, expected_game_value4_P1)
            l2 = min(expected_game_value2_P2, expected_game_value3_P2, expected_game_value4_P2)

            if l1 == expected_game_value4_P1:
                s = solutions4.sims["joint-0"].states[1]
                p0.first_progress = s["N↑"].x
                print("P1 Game 4")
            elif l1 == expected_game_value3_P1:
                s = solutions3.sims["joint-0"].states[1]
                p0.first_progress = s["level-0"].x
                print("P1 Game 3")
            elif l1 == expected_game_value2_P1:
                s = solutions2.sims["joint-0"].states[1]
                p0.first_progress = s["level-1"].x
                print("P1 Game 2")
            else:
                print("Something went really wrong!")

            if l2 == expected_game_value4_P2:
                s = solutions4.sims["joint-0"].states[1]
                p0.second_progress = s["W←"].x
                print("P2 Game 4")
            elif l2 == expected_game_value3_P2:
                s = solutions3.sims["joint-0"].states[1]
                p0.second_progress = s["level-1"].x
                print("P2 Game 3")
            elif l2 == expected_game_value2_P2:
                s = solutions2.sims["joint-0"].states[1]
                p0.second_progress = s["level-0"].x
                print("P2 Game 2")
            else:
                print("Something went really wrong 2!")

            # prob1.append(probab1)
            # prob2.append(probab2)
            #
            #
            # max_key1 = max(probab1, key=probab1.get)
            # max_key2 = max(probab2, key=probab2.get)
            #
            # if max_key1 == "Game_Normal":
            #     game4_preprocessed = preprocess_game(game4, solver_spec.solver_params)
            #     solutions4 = solve1(game4_preprocessed)
            #
            #     s = solutions4.sims["joint-0"].states[1]
            #     p0.first_progress = s['N↑'].x
            #     p0.second_progress = s['W←'].x
            #
            #     r1.append(s['N↑'])
            #     r2.append(s['W←'])
            #
            # elif max_key1 == "Game_MS":
            #     game2_preprocessed = preprocess_game(game2, solver_spec.solver_params)
            #     solutions2 = solve1(game2_preprocessed)
            #
            #     s = solutions2.sims["joint-0"].states[1]
            #     p0.first_progress = s['level-1'].x
            #     p0.second_progress = s['level-0'].x
            #
            #     r1.append(s['level-1'])
            #     r2.append(s['level-0'])
            #
            # elif max_key1 == "Game_MS":
            #     game3_preprocessed = preprocess_game(game3, solver_spec.solver_params)
            #     solutions3 = solve1(game3_preprocessed)
            #
            #     s = solutions3.sims["joint-0"].states[1]
            #     p0.first_progress = s['level-0'].x
            #     p0.second_progress = s['level-1'].x
            #
            #     r1.append(s['level-0'])
            #     r2.append(s['level-1'])
            #
            # else:
            #     game3_preprocessed = preprocess_game(game3, solver_spec.solver_params)
            #     solutions3 = solve1(game3_preprocessed)
            #
            #     s = solutions3.sims["joint-0"].states[1]
            #     p0.first_progress = s['level-0'].x
            #     p0.second_progress = s['level-1'].x
            #
            #     r1.append(s['level-0'])
            #     r2.append(s['level-1'])

            # Weigh each game's value with its probability
            # weighted_r11 = probab1["Game_Alone"] * r1
            # weighted_r12 = probab1["Game_MS"] * r2
            # weighted_r13 = probab1["Game_SM"] * r3
            # weighted_r14 = probab1["Game_Normal"] * r4
            # weighted_r1 = [weighted_r11, weighted_r12, weighted_r13, weighted_r14]
            #
            # weighted_r21 = probab2["Game_Alone"] * r1
            # weighted_r22 = probab2["Game_MS"] * r2
            # weighted_r23 = probab2["Game_SM"] * r3
            # weighted_r24 = probab2["Game_Normal"] * r4
            # weighted_r2 = [weighted_r21, weighted_r22, weighted_r23, weighted_r24]

            # # TODO Update step, solution states are needed. The idea is to implement the first step of the strategy with the most expected value.
            # # This is only a "dummy update" at the moment, it pushes the car one step further no matter what. Has to be changed.
            # if max(weighted_r1) == weighted_r11:
            #     p0.first_progress = p0.first_progress + 1
            # elif max(weighted_r1) == weighted_r12:
            #     p0.first_progress = p0.first_progress + 1
            # elif max(weighted_r1) == weighted_r13:
            #     p0.first_progress = p0.first_progress + 1
            # elif max(weighted_r1) == weighted_r14:
            #     p0.first_progress = p0.first_progress + 1
            # else:
            #     print("Something went wrong in the state update of car 1 of the belief game algorithm")
            #
            # if max(weighted_r2) == weighted_r21:
            #     p0.second_progress = p0.second_progress + 2
            # elif max(weighted_r2) == weighted_r22:
            #     p0.second_progress = p0.second_progress + 2
            # elif max(weighted_r2) == weighted_r23:
            #     p0.second_progress = p0.second_progress + 2
            # elif max(weighted_r2) == weighted_r24:
            #     p0.second_progress = p0.second_progress + 2
            # else:
            #     print("Something went wrong in the state update of car 2 of the belief game algorithm")

            # Use the new positions to determine whether they see each other now.
            seen = obs.do_we_see_us(p0.first_progress, p0.second_progress)

    print("end!!")

    # print("r1 = ", r1)
    # print("r2 = ", r2)
    # print("probab1 = ", prob1)
    # print("probab2 = ", prob2)
