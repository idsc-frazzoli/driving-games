from decimal import Decimal as D
from os.path import join

from belief_games import get_leader_follower_game, get_asym, get_asym_lf, TwoVehicleSeenObservation
from driving_games import get_alone, NO_LIGHTS, get_two_vehicle_game, TwoVehicleSimpleParams
from games import preprocess_game, solve1, SolverParams, STRATEGY_SECURITY, STRATEGY_MIX, report_solutions, \
    create_report_preprocessed
from games_scripts.solvers import SolverSpec



def test1():

    d = "ml_out/tests/"
    game_spec = get_asym_lf()
    game1 = game_spec.game
    game_name = "test_ml"
    solver_spec = SolverSpec("test", SolverParams(D(1), STRATEGY_MIX, False))
    solver_name = solver_spec.desc
    game_preprocessed = preprocess_game(game1, solver_spec.solver_params)
    solutions = solve1(game_preprocessed)
    for state, solution in solutions.game_solution.states_to_solution.items():
        # filter out only the first level subgame
        if all([p.stage == 1 for p in state.values()]):
            game_idx, _, _ = game.joint_reward.get_payoff_matrix_idx(state[p1_name], state[p2_name])
            print("Game solution of game:", _gamemat2str(leaves_payoffs[game_idx]))
            print("Joint state:\n", state)
            print("Values and actions:\n", solution.solved)
            print("Game values:\n", solution.va.game_value)
    d = "ml_out/tests/"
    dg = join(d, game_name)
    ds = join(dg, solver_name)
    r_solutions = report_solutions(game_preprocessed, solutions)
    r_preprocessed = create_report_preprocessed(game_name, game_preprocessed)

    r_solutions.to_html(join(ds, "r_solutions.html"))
    r_preprocessed.to_html(join(ds, "r_preprocessed.html"))

    print("end!!")