from typing import Optional, Callable, Mapping, Union
from dataclasses import dataclass

from games.solve.solution_structures import SolverParams, GameFactorization, GamePlayerPreprocessed
from games_zoo.solvers import SolverSpec
from games.performance import GetFactorizationPI
from games.game_def import (
    Game,
    PlayerName,
    RJ,
    RP,
    SR,
    U,
    X,
    Y
)


Player_Preprocessed = Mapping[PlayerName, GamePlayerPreprocessed[X, U, Y, RP, RJ, SR]]


@dataclass(frozen=True)
class FactorizationSolverParams(SolverParams):
    get_factorization: Optional[
        Callable[
            [Game[X, U, Y, RP, RJ, SR], SolverParams, Player_Preprocessed, Optional[GetFactorizationPI]],
            GameFactorization
        ]
    ]
    """ Contains the factorization algo used """
    beta: float
    """ 
    Standard deviation of gaussian filter applied to the solution strategy vector.
    Signifies the trust in the solution of a game.
    beta=0: full trust, resources of game
    beta=inf: no trust resources of forward reachable set
    """


@dataclass
class FactorizationSolverSpec(SolverSpec):
    solver_params: FactorizationSolverParams
    """ Contains the Factorization Solver Parameters """
