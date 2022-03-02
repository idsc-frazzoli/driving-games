from itertools import combinations
from typing import List, Dict
from dg_commons import PlayerName
from .parameters import player_idx


class Homotopy:
    players: List[PlayerName]
    homo_class: Dict[PlayerName, Dict[PlayerName, int]]
    def __init__(self, h):
        n_player

def evaluate_homotopy(intersects, vx_ref, players):
    """enumerate all homotopy classes and rank them"""



def heuristic_shortesttime(h, intersects, vx_ref, players):
    """
    a heuristic function to evaluate a homotopy class
     by computing the total time it takes for all vehicles to reach the intersection
    """
    total_time = 0
    for i_idx, player_pair in enumerate(combinations(players, 2)):
        player1 = player_pair[0]
        player2 = player_pair[1]
        if player2 in intersects[player1].keys():
            s1 = intersects[player1][player2]
            s2 = intersects[player2][player1]
        else:
            s1 = 0
            s2 = 0
        t1 = s1 / vx_ref[player_idx[player1]]
        t2 = s2 / vx_ref[player_idx[player2]]
        if h[player1][player2] == 0:  # player1 goes first
            if t1 < t2:  # no waiting time
                total_time += t1 + t2
            else:  # player2 has to wait for player1
                total_time += 2 * t1
        else:
            if t1 > t2:
                total_time += t1 + t2
            else:
                total_time += 2 * t2
    return total_time
