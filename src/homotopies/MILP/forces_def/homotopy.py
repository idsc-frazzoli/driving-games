from itertools import combinations, product
from typing import List, Dict
from dg_commons import PlayerName
from .parameters import player_idx


class Homotopy:
    def __init__(self, intersects, players, vx_ref=None, h=None):
        """
        vx_ref: vector of length = #player
        h: vector of length = #intersection
        """
        self.intersects = intersects
        self.players = players
        self.homo_class = {}
        n_inter = 0
        for player_pair in combinations(players, 2):
            player1 = player_pair[0]
            player2 = player_pair[1]
            if player2 not in intersects[player1].keys():  # player1 and player2 don't intersect
                continue
            if player1 not in self.homo_class.keys():
                self.homo_class[player1] = {}
            if h is None:
                self.homo_class[player1][player2] = 0
            else:
                self.homo_class[player1][player2] = h[n_inter]
            n_inter += 1
        if vx_ref is not None:
            self.heuristic = self.heuristic_shortesttime(vx_ref)
        else:
            self.heuristic = None

    def __lt__(self, other):
        return self.heuristic < other.heuristic

    def get_homotopy(self, player1, player2):
        if player1 in self.homo_class.keys():
            return self.homo_class[player1][player2]
        else:
            return self.homo_class[player2][player1]

    def set_homotopy(self, player1, player2, h):
        #  pay attention to the sequence of player1 and player2
        if player1 not in self.homo_class.keys() and player2 not in self.homo_class.keys():
            self.homo_class[player1] = {}
        if player1 in self.homo_class.keys():
            self.homo_class[player1][player2] = h
        else:
            self.homo_class[player2][player1] = h

    def heuristic_shortesttime(self, vx_ref):
        """
        a heuristic function to evaluate a homotopy class
         by computing the total time it takes for all vehicles to reach the intersection
         vx_ref: vector with length = #player
        """
        total_time = 0
        for player_pair in combinations(self.players, 2):
            player1 = player_pair[0]
            player2 = player_pair[1]
            if player2 not in self.intersects[player1].keys():  # player1 and player2 don't intersect
                continue
            s1 = self.intersects[player1][player2]
            s2 = self.intersects[player2][player1]
            t1 = s1 / vx_ref[player_idx[player1]]
            t2 = s2 / vx_ref[player_idx[player2]]
            h = self.get_homotopy(player1, player2)
            if h == 0:  # player1 goes first
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


def evaluate_homotopy(intersects, players, vx_ref):
    """enumerate all homotopy classes and rank them"""
    homotopies = []
    n_inter = 0
    for player_pair in combinations(players, 2):
        player1 = player_pair[0]
        player2 = player_pair[1]
        if player2 not in intersects[player1].keys():  # player1 and player2 don't intersect
            continue
        n_inter += 1
    for h in product([0, 1], repeat=n_inter):  # enumerate all homotopy classes
        homotopy = Homotopy(intersects, players, vx_ref, h)
        homotopies += [homotopy]
    homotopies_sorted = sorted(homotopies)
    return homotopies_sorted

