from dataclasses import dataclass
from decimal import Decimal as D
from typing import Dict

from games import SolverParams, STRATEGY_MIX, STRATEGY_SECURITY

__all__ = ["solvers_zoo"]


@dataclass
class SolverSpec:
    desc: str
    solver_params: SolverParams


solvers_zoo: Dict[str, SolverSpec] = {}

# The solution parameters
for strategy in [STRATEGY_MIX, STRATEGY_SECURITY]:
    params = SolverParams(dt=D(1.0), strategy_multiple_nash=strategy)
    solvers_zoo[f"solver-1-{strategy}"] = SolverSpec("discretization = 1", params)
#
# solvers_zoo["solver0.5"] = SolverSpec(
#     "discretization = 0.5", SolverParams(dt=D(0.5), strategy_multiple_nash=STRATEGY_SECURITY)
# )
