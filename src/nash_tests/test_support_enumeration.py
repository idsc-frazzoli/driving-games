from nashpy import support_enumeration as nashpy_sup_enum, vertex_enumeration
from nose.tools import assert_equal
from parameterized import parameterized

from nash.famous_games_zoo import matching_pennies, matching_pennies_2, BiMatGame, MINIMIZER, degenerate_1
from nash_tests import logger
from nash import ne_support_enum

games_to_test = [(matching_pennies,), (matching_pennies_2,), ]


@parameterized(games_to_test)
def test_bimatrix_games(game: BiMatGame) -> None:
    _A = game.A if game.p1_type == MINIMIZER else -game.A
    _B = game.B if game.p2_type == MINIMIZER else -game.B
    _eq = list(ne_support_enum(_A, _B))
    _equilibria = tuple(_eq)
    assert_equal(len(game.equilibria), len(_equilibria), msg="Different number of NE found")
    logger.info("found {}".format(len(_equilibria)))
    assert_equal(game.equilibria, _equilibria)


def test_single_game(game: BiMatGame = degenerate_1) -> None:
    """
    """
    _A = game.A if game.p1_type == MINIMIZER else -game.A
    _B = game.B if game.p2_type == MINIMIZER else -game.B
    logger.info(expected=game.equilibria)
    for i, eq in enumerate(ne_support_enum(_A, _B)):
        logger.info("My solver: NE ({}): ".format(i+1), eq)
    for i, eq in enumerate(nashpy_sup_enum(-_A, -_B, tol=0)):
        logger.info("Nashpy supp enum: NE ({}): ".format(i+1), eq)
    for i, eq in enumerate(vertex_enumeration(-_A, -_B)):
        logger.info("Nashpy vertex enum: NE ({}): ".format(i+1), eq)
