from dataclasses import dataclass
from decimal import Decimal as D

@dataclass
class SolverParams:
    dt: D
