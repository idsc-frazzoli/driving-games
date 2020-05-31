from dataclasses import replace
from typing import Mapping as M

from frozendict import frozendict

from .game_def import Game, PlayerName, RJ, RP, SR, U, X, Y


def get_individual_games(game: Game[X, U, Y, RP, RJ, SR]) -> M[PlayerName, Game[X, U, Y, RP, RJ, SR]]:
    """ Returns the individual games (by removing all others players)"""
    res = {}
    for player_name, player in game.players.items():
        players_only = frozendict({player_name: player})
        g = replace(game, players=players_only)
        res[player_name] = g
    return frozendict(res)
