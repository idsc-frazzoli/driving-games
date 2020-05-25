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
    dt = D(1.0)
    for n, use_factorization in [("fact", True), ("naive", False)]:
        params = SolverParams(
            dt=dt, strategy_multiple_nash=strategy, use_factorization=use_factorization
        )  # XXX
        desc = f"discretization = {dt}; factorization = {use_factorization}"
        solvers_zoo[f"solver-1-{strategy}-{n}"] = SolverSpec(desc, params)
    #
# solvers_zoo["solver0.5"] = SolverSpec(
#     "discretization = 0.5", SolverParams(dt=D(0.5), strategy_multiple_nash=STRATEGY_SECURITY)
# )
