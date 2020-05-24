from dataclasses import replace
from typing import Dict, Mapping

from frozendict import frozendict

from zuper_typing import debug_print
from zuper_typing.debug_print_ import eop
from . import logger
from .access import preprocess_player
from .game_def import (
    Game,
    PlayerName,
    Pr,
    RJ,
    RP,
    SR,
    U,
    X,
    Y,
)
from .solution import solve_game
from .structures_solution import GamePreprocessed, GameSolution, SolverParams

__all__ = ["compute_individual_solutions"]


def long_print(x: object) -> str:
    return debug_print(x, eop)


def compute_individual_solutions(
    game: Game[Pr, X, U, Y, RP, RJ, SR], solver_params: SolverParams
) -> Mapping[PlayerName, Mapping[X, GameSolution[Pr, X, U, Y, RP, RJ]]]:
    """ Computes the solutions that are individual """
    res: Dict[PlayerName, Mapping[X, GameSolution[Pr, X, U, Y, RP, RJ]]] = {}
    games = get_individual_games(game)
    for player_name, player_game in games.items():
        # game_graph = get_game_graph(player_game, dt=solver_params.dt)
        # compute_graph_layout(game_graph, iterations=1)
        player = player_game.players[player_name]
        player_pre = preprocess_player(game=game, player_name=player_name, player=player, dt=solver_params.dt)
        players_pre = {player_name: player_pre}
        
        gp = GamePreprocessed(
            game=player_game, players_pre=players_pre, game_graph=None, solver_params=solver_params,
            game_factorization=None
        )
        pres: Dict[X, GameSolution[Pr, X, U, Y, RP, RJ]] = {}
        for x0, personal_tree in player_pre.alone_tree.items():
            solved_x0: GameSolution[Pr, X, U, Y, RP, RJ] = solve_game(gp, personal_tree)
            pres[x0] = solved_x0
        res[player_name] = frozendict(pres)

    # logger.info(res=long_print(res))
    return frozendict(res)


def get_individual_games(
    game: Game[Pr, X, U, Y, RP, RJ, SR]
) -> Dict[PlayerName, Game[Pr, X, U, Y, RP, RJ, SR]]:
    """ Returns the individual games (by removing all others players)"""
    res = {}
    for player_name, player in game.players.items():
        players_only = frozendict({player_name: player})
        g = replace(game, players=players_only)
        res[player_name] = g
    return res
