import itertools
from fractions import Fraction
from itertools import product
from typing import Any, FrozenSet, Mapping, Sequence, Tuple, Type, List, Set
import numpy as np
from frozendict import frozendict
from decimal import Decimal as D
from zuper_commons.types import check_isinstance

from bayesian_driving_games import PlayerType
from games import JointRewardStructure, PlayerName, PersonalRewardStructure
from nash import BiMatGame
from toy_games.bayesian_toy_structures import BayesianBirdActions
from toy_games.toy_structures import BirdState, BirdCosts, BirdActions
from toy_games.bayesian_toy_structures import BayesianBirdState


class BayesianBirdJointReward(JointRewardStructure[BirdState, BirdActions, Any]):
    max_stages: int
    leaves_payoffs: Mapping[BirdState, Mapping[PlayerName, BirdCosts]]
    row_player: PlayerName
    col_player: PlayerName
    mat_payoffs: Sequence[np.ndarray]

    def __init__(
        self,
        max_stages: int,
        subgames: Sequence[BiMatGame],
        row_player: PlayerName,
        col_player: PlayerName,
        p1_types,
        p2_types,
    ):
        self.row_player = row_player
        self.col_player = col_player
        self.max_stages = max_stages
        assert len(subgames) == 8 or 4 or 1, subgames
        self.mat_payoffs = [np.stack([g.A, g.B], axis=-1) for g in subgames]
        self.p1_types = p1_types
        self.p2_types = p2_types

    def is_joint_final_state(self, xs: Mapping[PlayerName, BirdState]) -> FrozenSet[PlayerName]:
        res = set()
        if len(xs.items()) > 1:
            for player, x in xs.items():
                if x.stage == self.max_stages:
                    res.add(player)
        return frozenset(res)

    def joint_reward(
        self, xs: Mapping[PlayerName, BirdState]
    ) -> Mapping[Set[PlayerType], Mapping[PlayerName, BirdCosts]]:
        """
        Each payoff matrix correspond to a specific game
        [-1,-1] -> leaves_payoffs[0]
        [-1,1] -> leaves_payoffs[1]
        [1,-1] -> leaves_payoffs[2]
        [1,1] -> leaves_payoffs[3]
        :param xs:
        :return:
        """
        type_combinations = list(itertools.product(self.p1_types, self.p2_types))
        res1 = {}
        res2 = {}
        res = {}
        x1, x2 = xs[self.row_player], xs[self.col_player]
        subgame1, subgame2, row, col = self.get_payoff_matrix_idx(x1, x2)
        payoff11, payoff12 = self.mat_payoffs[subgame1][row, col, :]
        payoff21, payoff22 = self.mat_payoffs[subgame2][row, col, :]
        res1.update(
            {self.row_player: BirdCosts(D(payoff21.item())), self.col_player: BirdCosts(D(payoff22.item()))}
        )
        res2.update(
            {self.row_player: BirdCosts(D(payoff11.item())), self.col_player: BirdCosts(D(payoff12.item()))}
        )
        res[type_combinations[0]] = res1
        res[type_combinations[1]] = res2

        return frozendict(res)

    @staticmethod
    def get_payoff_matrix_idx(x1: BirdState, x2: BirdState) -> Tuple[int, int, int, int]:
        """
        Trick to build encapsulate payoff matrices into the dynamic game
        joint state (x1,x2):
        (<0,<0) -> G1
        (<0,>0) -> G2
        (>0,<0) -> G3
        (>0,>0) -> G4
        To figure out the indices we then look at the decimals...
        :param x1:
        :param x2:
        :return:
        """

        subgame1: int
        subgame2: int
        row: int
        col: int
        thresh = 0

        # Uncomment/Comment the following lines to switch from 1 stage to 2 stages.

        if x1.z < thresh and x2.z < thresh:
            subgame1 = 0
            subgame2 = 4
        elif x1.z < thresh < x2.z:
            subgame1 = 1
            subgame2 = 5
        elif x1.z > thresh > x2.z:
            subgame1 = 2
            subgame2 = 6
        else:
            subgame1 = 3
            subgame2 = 7
        # subgame = 0

        z1_dec = x1.z - round(x1.z)
        z2_dec = x2.z - round(x2.z)
        # z1_dec = x1.z
        # z2_dec = x2.z

        row, col = map(lambda x: 0 if x < 0 else 1, [z1_dec, z2_dec])

        return subgame1, subgame2, row, col
