import numpy as np

from nash import BiMatGame
from toy_games.toy_structures import ToyGameMat

G1 = BiMatGame(A=np.array([[1, 1], [3, 5]]), B=np.array([[0, 2], [2, 5]]))
G2 = BiMatGame(A=np.array([[1, 4], [2, 2]]), B=np.array([[0, 1], [3, 1]]))
G3 = BiMatGame(A=np.array([[1, 3], [2, 2]]), B=np.array([[4, 1], [1, 2]]))
G4 = BiMatGame(A=np.array([[3, 4], [3, 5]]), B=np.array([[3, 5], [2, 3]]))
game1 = ToyGameMat(subgames=[G1, G2, G3, G4], desc="This game presents some corner cases")

G1b = BiMatGame(A=np.array([[2, 4], [6, 8]]), B=np.array([[0, 1], [6, 5]]))
G2b = BiMatGame(A=np.array([[6, 4], [8, 6]]), B=np.array([[3, 4], [7, 9]]))
G3b = BiMatGame(A=np.array([[4, 8], [8, 4]]), B=np.array([[5, 3], [7, 3]]))
G4b = BiMatGame(A=np.array([[4, 10], [4, 8]]), B=np.array([[3, 3], [5, 3]]))
game2 = ToyGameMat(
    subgames=[G1b, G2b, G3b, G4b],
    desc="This game has a unique advantageous solution (NE).\n" "It is obtained by the outcome (2,0) in G1b.",
)

single_game = ToyGameMat(subgames=[G1b], desc="Single stage game")
