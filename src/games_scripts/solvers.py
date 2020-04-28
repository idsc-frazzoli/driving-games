from dataclasses import dataclass
from decimal import Decimal as D
from typing import Dict

from games import SolverParams

__all__ = ["solvers_zoo"]


@dataclass
class SolverSpec:
    desc: str
    solver_params: SolverParams


solvers_zoo: Dict[str, SolverSpec] = {}

# The solution parameters
solvers_zoo["solver1"] = SolverSpec("discretization = 1", SolverParams(dt=D(1.0)))

solvers_zoo["solver0.5"] = SolverSpec("discretization = 0.5", SolverParams(dt=D(0.5)))
