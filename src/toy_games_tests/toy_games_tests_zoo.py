from dataclasses import dataclass
from typing import List

import numpy as np

from nash import BiMatGame



@dataclass
class ToyGame:
    subgames: List[BiMatGame]
    desc: str

    def __post_init__(self):
        assert len(self.subgames) == 4, len(self.subgames)


G1 = BiMatGame(A=np.array([[1, 1], [3, 5]]), B=np.array([[0, 2], [2, 5]]))
G2 = BiMatGame(A=np.array([[1, 4], [2, 2]]), B=np.array([[0, 1], [3, 1]]))
G3 = BiMatGame(A=np.array([[1, 3], [2, 2]]), B=np.array([[4, 1], [1, 2]]))
G4 = BiMatGame(A=np.array([[3, 4], [3, 5]]), B=np.array([[3, 5], [2, 3]]))
game1 = ToyGame(subgames=[G1, G2, G3, G4], desc="This game presents some corner cases")

G1b = BiMatGame(A=np.array([[1, 4], [6, 5]]), B=np.array([[0, 2], [6, 5]]))
G2b = BiMatGame(A=np.array([[5, 4], [8, 6]]), B=np.array([[5, 4], [8, 6]]))
G3b = BiMatGame(A=np.array([[4, 3], [9, 3]]), B=np.array([[4, 3], [9, 3]]))
G4b = BiMatGame(A=np.array([[3, 3], [3, 8]]), B=np.array([[3, 3], [2, 0]]))
game2 = ToyGame(subgames=[G1b, G2b, G3b, G4b], desc="This game shall have a unique clear solution")
