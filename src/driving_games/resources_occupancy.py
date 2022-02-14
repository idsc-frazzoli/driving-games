from dataclasses import dataclass
from decimal import Decimal
from typing import FrozenSet, Tuple

CellID = Tuple[int, int]


@dataclass
class ResourcesOccupancy:
    cell_resolution: Decimal

    def get_occupied_cells(
        self,
    ) -> FrozenSet[CellID]:
        pass
