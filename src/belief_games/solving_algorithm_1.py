from decimal import Decimal as D
from os.path import join

from belief_games import get_master_slave_game, get_asym, TwoVehicleSeenObservation
from belief_games.belief_calculation import belief_calc_simple1, belief_calc_simple2
from driving_games import NO_LIGHTS, get_two_vehicle_game, TwoVehicleSimpleParams, PlayerName
from games import preprocess_game, solve1, SolverParams, STRATEGY_SECURITY, STRATEGY_MIX, report_solutions, \
    create_report_preprocessed
from games.get_indiv_games import get_individual_games
from games_scripts.solvers import SolverSpec


def solving_hidden_agent_game():

    # Define Parameters
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

    # starting positons of the two cars
    start = p0.side + p0.road_lane_offset
    L = p0.side + p0.road + p0.side

    p1_ref = (D(start), D(0), D(+90))
    p2_ref = (D(L), D(start), D(-180))

    # See if the two cars initially see each other
    obs = TwoVehicleSeenObservation(road, side, p1_ref, p2_ref)
    seen = obs.do_we_see_us(p0.first_progress, p0.second_progress)

    # Loop starts here
    while True:

        # Every timestep, determine whether the cars see each other. If yes, play a normal intersection game and finish.
        if seen:
            game_seen = get_two_vehicle_game(p0)
            solver_spec = SolverSpec("test", SolverParams(D(1), STRATEGY_MIX, False))
            solver_name = solver_spec.desc
            game_seen_preprocessed = preprocess_game(game_seen, solver_spec.solver_params)
            solutions = solve1(game_seen_preprocessed)
            break

        # If no, do the following
        else:
            # Define the four games
            game_seen = get_two_vehicle_game(p0)
            res = get_individual_games(game_seen)
            alone_player = list(res.keys())[0]
            game1 = res[alone_player] #Alone game
            game2 = get_master_slave_game(p0, True) # principal-subordinate This game is the same as the two_vehicle_game, but in the
            # preferences, "Ignore_second" is set to True.
            game3 = get_master_slave_game(p0, False) # Same as above. The parameter in the function call determines
            # if car 1 or car 2 is the master
            game4 = game_seen #Normal two vehicle game

            # Alone game
            solver_spec = SolverSpec("test", SolverParams(D(1), STRATEGY_MIX, False))
            solver_name = solver_spec.desc

            # Preprocess and solve games. The alone game ("Game1") gives an error at the moment, because when
            # creating the game graph, the number of players is hardcoded to 2 at the moment...
            game1_preprocessed = preprocess_game(game1, solver_spec.solver_params)
            solutions1 = solve1(game1_preprocessed)
            game2_preprocessed = preprocess_game(game2, solver_spec.solver_params)
            solutions2 = solve1(game2_preprocessed)
            game3_preprocessed = preprocess_game(game3, solver_spec.solver_params)
            solutions3 = solve1(game3_preprocessed)
            game4_preprocessed = preprocess_game(game4, solver_spec.solver_params)
            solutions4 = solve1(game4_preprocessed)

            #######
            # TODO Values of the game. In this section, we should try to access the value/Nash eq. of the game in order to compare them...
            r1, r2, r3, r4 = [], [], [], []

            # My try of getting a hand on any values of the game => must be changed.
            for joint_state, node in solutions1.game_solution.states_to_solution.items():
                print(list(solutions1.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0])
                if list(solutions1.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0].joint.active == False:
                    r1.append(list(solutions1.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0].personal.duration)
                else:
                    pass
                if list(solutions2.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0].joint.active == False:
                    r2 = r2.append(list(solutions2.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0].personal.duration)
                else:
                    pass
                if list(solutions3.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0].joint.active == False:
                    r3 = r3.append(list(solutions3.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0].personal.duration)
                else:
                    pass
                if list(solutions4.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0].joint.active == False:
                    r4 = r4.append(list(solutions4.game_solution.states_to_solution[joint_state].va.game_value["Caring"].support())[0].personal.duration)
                else:
                    pass
                ######

            # calculate the probabilities of playing each game.
            probab1 = belief_calc_simple1(p0)
            probab2 = belief_calc_simple2(p0)

            # Weigh each game's value with its probability
            weighted_r11 = probab1["Game_Alone"] * r1
            weighted_r12 = probab1["Game_MS"] * r2
            weighted_r13 = probab1["Game_SM"] * r3
            weighted_r14 = probab1["Game_Normal"] * r4
            weighted_r1 = [weighted_r11, weighted_r12, weighted_r13, weighted_r14]

            weighted_r21 = probab2["Game_Alone"] * r1
            weighted_r22 = probab2["Game_MS"] * r2
            weighted_r23 = probab2["Game_SM"] * r3
            weighted_r24 = probab2["Game_Normal"] * r4
            weighted_r2 = [weighted_r21, weighted_r22, weighted_r23, weighted_r24]

            # TODO Update step, solution states are needed. The idea is to implement the first step of the strategy with the most expected value.
            # This is only a "dummy update" at the moment, it pushes the car one step further no matter what. Has to be changed.
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

            # Use the new positions to determine whether they see each other now.
            seen = obs.do_we_see_us(p0.first_progress, p0.second_progress)

    print("end!!")