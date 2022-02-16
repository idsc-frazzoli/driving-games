from dataclasses import dataclass
from decimal import Decimal as D
from itertools import product
from typing import Dict

from games import FINITE_MIX_STRATEGIES, MIX_MNE, PURE_STRATEGIES, SECURITY_MNE, SolverParams
from games.factorization_algo import FactAlgoNoFact, FactAlgoReachableRes
from .resources_occupancy import cells_resources_checker

__all__ = ["solvers_zoo"]


@dataclass
class SolverSpec:
    desc: str
    solver_params: SolverParams


solvers_zoo: Dict[str, SolverSpec] = {}

# The solution parameters
# fixme review the documentation
admissible_strategies = [PURE_STRATEGIES, FINITE_MIX_STRATEGIES]
mne_strategies = [MIX_MNE, SECURITY_MNE]
discretization_steps = [
    D(2.0),
    D(1.0),
]
fact_options = [("fact", True, FactAlgoReachableRes), ("naive", False, FactAlgoNoFact)]
extra_options = [("extra", True), ("noextra", False)]

options_mix = [admissible_strategies, mne_strategies, discretization_steps, fact_options, extra_options]

for adm_strat, mne_strat, dt, fact, extra in product(*options_mix):
    params = SolverParams(
        dt=dt,
        admissible_strategies=adm_strat,
        strategy_multiple_nash=mne_strat,
        n_simulations=1,
        use_factorization=fact[1],
        factorization_algorithm=fact[2](cells_resources_checker),
        extra=extra[1],
    )
    desc = (
        f"Admissible strategies = {adm_strat}; Multiple NE strategy = {mne_strat}; "
        f"discretization = {dt}; factorization = {fact[2].__name__}; extra = {extra[1]}"
    )
    # todo: update with algo fact name
    solvers_zoo[f"solver-{dt}-{adm_strat}-{mne_strat}-{fact[0]}-{extra[0]}"] = SolverSpec(desc, params)
