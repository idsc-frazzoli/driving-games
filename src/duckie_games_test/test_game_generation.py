from duckie_games.game_generation import get_duckie_game

from duckie_games.zoo import (
    two_player_reference_game_parameters,
    uncertainty_sets,
    uncertainty_prob
)


def test_game_generation():
    duckie_game_sets = get_duckie_game(
        duckie_game_params=two_player_reference_game_parameters,
        uncertainty_params=uncertainty_sets
    )

    duckie_game_prop = get_duckie_game(
        duckie_game_params=two_player_reference_game_parameters,
        uncertainty_params=uncertainty_prob
    )
