from typing import Sequence

from _tmp.bayesian_driving_games.solution import solve_bayesian_game

from _tmp.bayesian_driving_games.preprocess import preprocess_bayesian_game
from _tmp.nash import print_bimatgame
from _tmp.toy_games import ToyGameMat, get_toy_game_spec

from _tmp.toy_games.bayesian_toy_game import get_bayesian_toy_game_spec

from driving_games import UncertaintyParams
from games import PlayerName, preprocess_game, solve1
from games_zoo.solvers import SolverSpec
from _tmp.nash import BiMatGame
from _tmp.toy_games.bayesian_toy_rewards import BayesianBirdJointReward
from _tmp.toy_games_tests import logger


def _run_toy_game(
    toy_game_mat: ToyGameMat,
    solver_spec: SolverSpec,
    uncertainty_params: UncertaintyParams,
):
    p1_name, p2_name = PlayerName("1"), PlayerName("2")

    logger.info(f"Starting a {toy_game_mat.get_max_stages()} stage toy game with the following subgame(s):")
    for i, bimatgame in enumerate(toy_game_mat.subgames):
        logger.info(f"Subgame {i}: {print_bimatgame(bimatgame)}")

    solver_params = solver_spec.solver_params
    game_spec = get_toy_game_spec(toy_game_mat, uncertainty_params)
    game = game_spec.game
    game_preprocessed = preprocess_game(game, solver_params)
    solutions = solve1(game_preprocessed)
    # for state, solution in solutions.game_solution.states_to_solution.items():
    #     # filter out only the first level subgame
    #     if all([p.stage == 1 for p in state.values()]):
    #         game_idx, _, _ = BirdJointReward.get_payoff_matrix_idx(
    #             toy_game_mat.get_max_stages(), state[p1_name], state[p2_name]
    #         )
    #         # print("Game solution of game:", gamemat2str(leaves_payoffs[game_idx]))
    #         logger.info("Joint state:\n", state)
    #         logger.info("Values and actions:\n", solution.solved)
    #         logger.info("Game values:\n", solution.va.game_value)

    # todo check solutions with what we expect
    # todo create report/visualisation
    # logger.info(solutions)


def _run_toy_game_bayesian(
    subgames: Sequence[BiMatGame],
    solver_spec: SolverSpec,
    uncertainty_params: UncertaintyParams,
):
    max_stages = 2
    p1_name, p2_name = PlayerName("1"), PlayerName("2")

    logger.info("Starting a 2 stage toy game with the following subgames:")
    for i, bimatgame in enumerate(subgames):
        logger.info("Subgame {}: {}".format(i, print_bimatgame(bimatgame)))

    solver_params = solver_spec.solver_params
    # game_spec = get_toy_game_spec(max_stages, subgames, uncertainty_params)
    game_spec = get_bayesian_toy_game_spec(max_stages, subgames, uncertainty_params)
    game = game_spec.game
    game_preprocessed = preprocess_bayesian_game(game, solver_params)
    solutions = solve_bayesian_game(game_preprocessed)
    for state, solution in solutions.game_solution.states_to_solution.items():
        # filter out only the first level subgame
        if all([p.stage == 1 for p in state._values()]):
            game_idx1, game_idx2, _, _ = BayesianBirdJointReward.get_payoff_matrix_idx(
                max_stages, state[p1_name], state[p2_name]
            )
            ##print("Game solution of game:", gamemat2str(leaves_payoffs[game_idx]))
            logger.info("Joint state:\n", state)
            logger.info("Values and actions:\n", solution.solved)
            logger.info("Game values:\n", solution.va.game_value)

    # todo create report/visualisation
    # logger.info(solutions)
