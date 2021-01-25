from itertools import product

from parameterized import parameterized

from duckie_games.game_generation import get_duckie_game

from duckie_games.zoo import (
    two_player_duckie_game_parameters,
    two_player_duckie_game_parameters_stretched,
    uncertainty_sets,
    uncertainty_prob
)

uncertainty_params = [
    uncertainty_sets,
    #uncertainty_prob,
]

duckie_game_params = [
    #two_player_duckie_game_parameters,
    two_player_duckie_game_parameters_stretched
]

params = list(product(duckie_game_params, uncertainty_params))

@parameterized(params)
def test_game_generation(game_params, uncertain_params):
    """
    Testing if the game params from duckiegames.zoo can be loaded with the game_loader
    """
    duckie_game_sets = get_duckie_game(
        duckie_game_params=game_params,
        uncertainty_params=uncertain_params
    )

