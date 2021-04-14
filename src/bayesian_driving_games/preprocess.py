from typing import Optional

from frozendict import frozendict

from bayesian_driving_games.create_joint_game_tree import create_bayesian_game_graph
from bayesian_driving_games.structures import BayesianGame
from bayesian_driving_games.structures_solution import BayesianGameGraph
from games import Game, X, U, Y, RP, RJ, SolverParams, GamePreprocessed
from games.access import get_game_graph, compute_graph_layout, get_game_factorization, get_player_graph
from games.game_def import SR
from games.get_indiv_games import get_individual_games
from games.utils import valmap
from bayesian_driving_games.solution import solve_game_bayesian2
from games.solve.solution_structures import GameFactorization, GamePlayerPreprocessed, GameSolution


def preprocess_bayesian_player(
    individual_game: BayesianGame,
    solver_params: SolverParams,
) -> GamePlayerPreprocessed[X, U, Y, RP, RJ, SR]:
    """
    Same logic as in a driving game, it uses the functions `create_bayesian_game_graph` and
    `solve_game_bayesian2` instead of the deterministic ones.

    :param individual_game:
    :param solver_params:
    :return: Preprocess for each player
    """
    l = list(individual_game.players)
    assert len(l) == 1
    player_name = l[0]
    player = individual_game.players[player_name]
    graph = get_player_graph(player, solver_params.dt)

    game_graph: BayesianGameGraph
    initials = frozenset(map(lambda x: frozendict({player_name: x}), player.initial.support()))

    game_graph = create_bayesian_game_graph(individual_game, solver_params.dt, initials, gf=None)

    gs: GameSolution[X, U, Y, RP, RJ, SR]
    gs = solve_game_bayesian2(game=individual_game, solver_params=solver_params, gg=game_graph, jss=initials)

    return GamePlayerPreprocessed(graph, game_graph, gs)


def preprocess_bayesian_game(
    game: Game[X, U, Y, RP, RJ, SR],
    solver_params: SolverParams,
) -> GamePreprocessed[X, U, Y, RP, RJ, SR]:
    # todo BayesianGamePreprocessed
    """
    Same as in Driving Games, except that it uses bayesian_preprocess_player.

    :param game:
    :param solver_params:
    :return:
    """
    game_factorization: Optional[GameFactorization[X]] = None

    game_graph = get_game_graph(game, dt=solver_params.dt)
    compute_graph_layout(game_graph, iterations=1)
    individual_games = get_individual_games(game)
    players_pre = valmap(
        lambda individual_game: preprocess_bayesian_player(
            solver_params=solver_params, individual_game=individual_game
        ),
        individual_games,
    )
    if solver_params.use_factorization:
        game_factorization = get_game_factorization(game, solver_params, players_pre)
    # todo BayesianGamePreprocessed
    gp = GamePreprocessed(
        game=game,
        players_pre=players_pre,
        game_graph=game_graph,
        solver_params=solver_params,
        game_factorization=game_factorization,
    )

    return gp
