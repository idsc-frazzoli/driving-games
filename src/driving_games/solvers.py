from dataclasses import dataclass
from decimal import Decimal as D
from itertools import product
from typing import Dict

from games import SolverParams, MIX_MNE, SECURITY_MNE, FINITE_MIX_STRATEGIES, PURE_STRATEGIES

__all__ = ["solvers_zoo"]


@dataclass
class SolverSpec:
    desc: str
    solver_params: SolverParams


solvers_zoo: Dict[str, SolverSpec] = {}

# The solution parameters
admissible_strategies = [PURE_STRATEGIES, FINITE_MIX_STRATEGIES]
mne_strategies = [MIX_MNE, SECURITY_MNE]
dts = [
    D(1.0),
]
fact_options = [("fact", True), ("naive", False)]

options_mix = [admissible_strategies, mne_strategies, dts, fact_options]

for adm_strat, mne_strat, dt, fact in product(*options_mix):
    params = SolverParams(
        dt=dt, admissible_strategies=adm_strat, strategy_multiple_nash=mne_strat, use_factorization=fact[1]
    )
    desc = (
        f"Admissible strategies = {adm_strat}; Multiple NE strategy = {mne_strat}; "
        f"discretization = {dt}; factorization = {fact[1]}"
    )
    solvers_zoo[f"solver-1-{adm_strat}-{mne_strat}-{fact[0]}"] = SolverSpec(desc, params)

#
# solvers_zoo["solver0.5"] = SolverSpec(
#     "discretization = 0.5", SolverParams(dt=D(0.5), strategy_multiple_nash=STRATEGY_SECURITY)
# )
