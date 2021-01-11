from typing import Optional

from toolz import valmap

from games.access import preprocess_player, get_game_graph, compute_graph_layout
from games.game_def import (
    Game,
    RJ,
    RP,
    SR,
    U,
    X,
    Y,
)
from games.get_indiv_games import get_individual_games
from games.solve.solution_structures import (
    GameFactorization,
    GamePreprocessed,
)

from .structures import FactorizationSolverParams


def preprocess_game_factorization(
    game: Game[X, U, Y, RP, RJ, SR],
    solver_params: FactorizationSolverParams,
) -> GamePreprocessed[X, U, Y, RP, RJ, SR]:
    """
    1. Preprocesses the game computing the general game graph (MultiDiGraph used for visualisation)
    2. Computes the solutions for the single players
    3. If factorization is selected, computes the corresponding game factorization

    :param game:
    :param solver_params:
    :return:
    """
    game_factorization: Optional[GameFactorization[X]] = None

    game_graph = get_game_graph(game, dt=solver_params.dt)
    compute_graph_layout(game_graph, iterations=1)
    individual_games = get_individual_games(game)
    players_pre = valmap(
        lambda individual_game: preprocess_player(
            solver_params=solver_params, individual_game=individual_game
        ),
        individual_games,
    )
    if solver_params.use_factorization:
        game_factorization = solver_params.get_factorization(game, players_pre)

    gp = GamePreprocessed(
        game=game,
        players_pre=players_pre,
        game_graph=game_graph,
        solver_params=solver_params,
        game_factorization=game_factorization,
    )

    return gp

