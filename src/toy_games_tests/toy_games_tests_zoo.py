from dataclasses import dataclass
from typing import List

import numpy as np


def gamemat2str(mat: np.ndarray) -> str:
    assert mat.shape[0] == 2
    rows, cols = mat.shape[1:3]
    str_game = ""
    for r in range(rows):
        str_game += "\n\t"
        for c in range(cols):
            str_game += "{},{}\t\t".format(mat[0, r, c], mat[1, r, c])
    return str_game


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
game1 = ToyGame(subgames=[G1, G2, G3, G4], desc="This game presents some corner cases")

G1b = np.array([[[1, 0], [4, 2]], [[5, 6], [5, 5]]])
G2b = np.array([[[2, 5], [4, 1]], [[2, 8], [2, 6]]])
G3b = np.array([[[2, 4], [5, 3]], [[2, 9], [2, 3]]])
G4b = np.array([[[9, 3], [2, 3]], [[3, 2], [8, 0]]])
game2 = ToyGame(subgames=[G1b, G2b, G3b, G4b], desc="This game shall have a unique clear solution")
