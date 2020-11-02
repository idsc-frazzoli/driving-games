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

G1b = BiMatGame(A=np.array([[2, 4], [6, 8]]), B=np.array([[0, 1], [6, 5]]))
G2b = BiMatGame(A=np.array([[6, 4], [8, 6]]), B=np.array([[3, 4], [7, 9]]))
G3b = BiMatGame(A=np.array([[4, 8], [8, 4]]), B=np.array([[5, 3], [7, 3]]))
G4b = BiMatGame(A=np.array([[4, 10], [4, 8]]), B=np.array([[3, 3], [5, 3]]))
game2 = ToyGame(
    subgames=[G1b, G2b, G3b, G4b],
    desc="This game has a unique advantageous solution (NE).\n" "It is obtained by the outcome (2,0) in G1b.",
)
G1b = BiMatGame(A=np.array([[100, 200], [300, 400]]), B=np.array([[100, 200], [300, 400]]))  # down down
G2b = BiMatGame(A=np.array([[500, 600], [700, 800]]), B=np.array([[500, 600], [700, 800]]))  # down up
G3b = BiMatGame(A=np.array([[900, 1000], [1100, 1200]]), B=np.array([[900, 1000], [1100, 1200]]))  # up down
G4b = BiMatGame(A=np.array([[1300, 1400], [1500, 1600]]), B=np.array([[1300, 1400], [1500, 1600]]))  # up up
game2 = ToyGame(
    subgames=[G1b, G2b, G3b, G4b],
    desc="This game should have a unique clear solution, but what do the weights do?",
)

G1c = BiMatGame(A=np.array([[1, 2], [3, 4]]), B=np.array([[1, 2], [3, 4]]))  # down down
G2c = BiMatGame(A=np.array([[1, 2], [3, 4]]), B=np.array([[1, 2], [3, 4]]))  # down up
G3c = BiMatGame(A=np.array([[1, 2], [3, 4]]), B=np.array([[1, 2], [3, 4]]))  # up down
G4c = BiMatGame(A=np.array([[1, 2], [3, 4]]), B=np.array([[1, 2], [3, 4]]))  # up up
game3 = ToyGame(subgames=[G1c, G2c, G3c, G4c], desc="Every subgame has a different clear unique equilibrium")


G1d = BiMatGame(A=np.array([[1, 2], [3, 4]]), B=np.array([[1, 2], [3, 4]]))  # down down
game4 = ToyGame(subgames=[G1d], desc="one-stage game")

G1e = BiMatGame(A=np.array([[1, 2], [3, 4]]), B=np.array([[1, 2], [3, 4]]))  # down down
G2e = BiMatGame(A=np.array([[5, 6], [7, 8]]), B=np.array([[5, 6], [7, 8]]))  # down up
G3e = BiMatGame(A=np.array([[9, 10], [11, 12]]), B=np.array([[9, 10], [11, 12]]))  # up down
G4e = BiMatGame(A=np.array([[13, 14], [15, 16]]), B=np.array([[13, 14], [15, 16]]))  # up up
game5 = ToyGame(subgames=[G1e, G2e, G3e, G4e], desc="All different outcomes")

G1f = BiMatGame(A=np.array([[-1, -2], [-2, -9]]), B=np.array([[-1, -2], [-2, -9]]))  # down down
G2f = BiMatGame(A=np.array([[-1, -1], [-1, -1]]), B=np.array([[-1, -1], [-1, -1]]))  # down up
G3f = BiMatGame(A=np.array([[-1, -1], [-1, -1]]), B=np.array([[-1, -1], [-1, -1]]))  # up down
G4f = BiMatGame(A=np.array([[-1, -2], [-2, -6]]), B=np.array([[-1, -2], [-2, -6]]))  # up up
game6 = ToyGame(subgames=[G1f, G2f, G3f, G4f], desc="Bayesian Game that should favour 6,6 instead of 9,9")


G1g = BiMatGame(A=np.array([[-20, -15], [-10, -5]]), B=np.array([[0, -1], [-5, 0]]))  # down down
G2g = BiMatGame(A=np.array([[-15, -12], [-10, -5]]), B=np.array([[-1, -2], [0, -3]]))  # down up
G3g = BiMatGame(A=np.array([[-10, -10], [0, 0]]), B=np.array([[-5, 0], [-10, -5]]))  # up down
G4g = BiMatGame(A=np.array([[-10, -5], [0, 0]]), B=np.array([[0, -1], [-2, 0]]))  # up up

G5g = BiMatGame(A=np.array([[0, 0], [-5, -5]]), B=np.array([[-10, 5], [-5, 0]]))  # down down
G6g = BiMatGame(A=np.array([[0, 0], [-10, -5]]), B=np.array([[-1, -2], [0, -3]]))  # down up
G7g = BiMatGame(A=np.array([[-5, -10], [-15, -15]]), B=np.array([[-5, 0], [-2, -1]]))  # up down
G8g = BiMatGame(A=np.array([[-10, -5], [-15, -20]]), B=np.array([[0, -1], [-5, 0]]))  # up up
game7 = ToyGame(subgames=[G1g, G2g, G3g, G4g, G5g, G6g, G7g, G8g], desc="A dominated Bayesian Game")
