from decimal import Decimal
from typing import Union, Tuple, NewType

__all__ = ["SimTime", "ImpactLocation", "Color"]

Color = Union[Tuple[float], str]
SimTime = Decimal
ImpactLocation = NewType("ImpactLocation", str)
