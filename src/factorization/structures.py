from typing import Optional, Callable, Mapping
from dataclasses import dataclass

from games.solve.solution_structures import SolverParams, GameFactorization, GamePlayerPreprocessed
from games_zoo.solvers import SolverSpec
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
    get_factorization: Optional[Callable[[Game[X, U, Y, RP, RJ, SR], Player_Preprocessed], GameFactorization]]
    """ Contains the factorization algo used """

@dataclass
class FactorizationSolverSpec(SolverSpec):
    solver_params: FactorizationSolverParams