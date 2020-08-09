from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class ToyGame:
    subgames: List[np.ndarray]
    desc: str

    def __post_init__(self):
        assert len(self.subgames) == 4, len(self.subgames)


G1 = np.array([[[1, 0], [1, 2]], [[3, 2], [5, 5]]])
G2 = np.array([[[1, 0], [4, 1]], [[2, 3], [2, 1]]])
G3 = np.array([[[1, 4], [3, 1]], [[2, 1], [2, 2]]])
G4 = np.array([[[3, 3], [4, 3]], [[3, 2], [5, 3]]])
#game1 = ToyGame(subgames=[G1, G2, G3, G4], desc="This game presents some corner cases")

G1b = np.array([[[1, 2], [3, 4]], [[2, 1], [3, 4]]])
G2b = np.array([[[5, 6], [7, 8]], [[5, 6], [7, 8]]])
G3b = np.array([[[9, 10], [11, 12]], [[9, 10], [11, 12]]])
G4b = np.array([[[13, 14], [15, -16]], [[13, 14], [15, -16]]])

game2 = ToyGame(subgames=[G1b, G2b, G3b, G4b], desc="This game shall have a unique clear solution")
game1 = game2