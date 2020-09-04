from dataclasses import dataclass
from typing import List

import numpy as np

from nash import BiMatGame


@dataclass
class ToyGame:
    subgames: List[BiMatGame]
    desc: str

    def __post_init__(self):
        assert len(self.subgames) == 4 or 1, len(self.subgames)


G1 = BiMatGame(A=np.array([[1, 1], [3, 5]]), B=np.array([[0, 2], [2, 5]]))
G2 = BiMatGame(A=np.array([[1, 4], [2, 2]]), B=np.array([[0, 1], [3, 1]]))
G3 = BiMatGame(A=np.array([[1, 3], [2, 2]]), B=np.array([[4, 1], [1, 2]]))
G4 = BiMatGame(A=np.array([[3, 4], [3, 5]]), B=np.array([[3, 5], [2, 3]]))
game1 = ToyGame(subgames=[G1, G2, G3, G4], desc="This game presents some corner cases")

G1b = BiMatGame(A=np.array([[1, 200], [300, 400]]), B=np.array([[1, 200], [300, 400]])) #down down
G2b = BiMatGame(A=np.array([[500, 600], [700, 800]]), B=np.array([[500, 600], [700, 800]])) #down up
G3b = BiMatGame(A=np.array([[900, 1000], [1100, 1200]]), B=np.array([[900, 1000], [1100, 1200]])) #up down
G4b = BiMatGame(A=np.array([[1300, 1400], [1500, 1600]]), B=np.array([[1300, 1400], [1500, 1600]])) #up up
game2 = ToyGame(subgames=[G1b, G2b, G3b, G4b], desc="This game should have a unique clear solution, but what do the weights do?")

G1c = BiMatGame(A=np.array([[-1, 2], [3, 4]]), B=np.array([[-1, 2], [3, 4]])) #down down
G2c = BiMatGame(A=np.array([[5, 6], [7, 8]]), B=np.array([[5, 6], [7, 8]])) #down up
G3c = BiMatGame(A=np.array([[9, 10], [11, 12]]), B=np.array([[9, 10], [11, 12]])) #up down
G4c = BiMatGame(A=np.array([[13, 14], [15, 16]]), B=np.array([[13, 14], [15, 16]])) #up up
game3 = ToyGame(subgames=[G1c, G2c, G3c, G4c], desc="This game should have a unique clear solution")


G1d = BiMatGame(A=np.array([[1, 2], [3, 4]]), B=np.array([[1, 2], [3, 4]])) #down down
game4 = ToyGame(subgames=[G1d], desc="one-stage game")