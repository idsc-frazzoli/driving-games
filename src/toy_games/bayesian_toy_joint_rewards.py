from itertools import product
from typing import Any, FrozenSet, Mapping, Sequence, Tuple, Type, List
import numpy as np
from frozendict import frozendict
from decimal import Decimal as D

from games import JointRewardStructure, PlayerName
from nash import BiMatGame
from toy_games.toy_structures import BirdState, BirdCosts, BirdActions, BayesianBirdState


class BayesianBirdJointReward(JointRewardStructure[BayesianBirdState, BirdActions, Any]):
    max_stages: int
    leaves_payoffs: Mapping[BayesianBirdState, Mapping[PlayerName, BirdCosts]]
    row_player: PlayerName
    col_player: PlayerName
    mat_payoffs: Sequence[np.ndarray]

    def __init__(
        self, max_stages: int, subgames: Sequence[BiMatGame], row_player: PlayerName, col_player: PlayerName,
            t1: Mapping[PlayerName, List], t2: Mapping[PlayerName, List]):
        self.row_player = row_player
        self.col_player = col_player
        self.max_stages = max_stages
        self.t1 = t1
        self.t2 = t2
        assert len(subgames) == 4 or 1, subgames
        self.mat_payoffs = [np.stack([g.A, g.B], axis=-1) for g in subgames]

    def is_joint_final_state(self, xs: Mapping[PlayerName, BirdState]) -> FrozenSet[PlayerName]:
        res = set()
        if len(xs.items()) > 1:
            for player, x in xs.items():
                if x.stage == self.max_stages:
                    res.add(player)
        return frozenset(res)

    def joint_reward(self, xs: Mapping[PlayerName, BayesianBirdState]) -> Mapping[PlayerName, BirdCosts]:
        """
        Each payoff matrix correspond to a specific game
        [-1,-1] -> leaves_payoffs[0]
        [-1,1] -> leaves_payoffs[1]
        [1,-1] -> leaves_payoffs[2]
        [1,1] -> leaves_payoffs[3]
        :param xs:
        :return:
        """
        res = {}
        x1, x2 = xs[self.row_player], xs[self.col_player]
        subgame, row, col = self.get_payoff_matrix_idx(x1, x2)
        payoff1, payoff2 = self.mat_payoffs[subgame][row, col, :]
        res.update(
            {self.row_player: BirdCosts(D(payoff1.item())), self.col_player: BirdCosts(D(payoff2.item()))}
        )
        return frozendict(res)

    @staticmethod
    def get_payoff_matrix_idx(x1: BayesianBirdState, x2: BayesianBirdState) -> Tuple[int, int, int]:
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

        subgame: int
        row: int
        col: int
        thresh = 0

        # type_combi = {**t1, **t2}
        # allKeys = sorted(type_combi)
        # combinations = product(*(type_combi[Name] for Name in allKeys))
        #
        # for combi in combinations:

        if (x1.t == ('aggressive')) & (x2.t == ('aggressive')):
            subgame = 0
        elif (x1.t == ('aggressive')) & (x2.t == ('cautious')):
            subgame = 1
        elif (x1.t == ('cautious')) & (x2.t == ('aggressive')):
            subgame = 2
        elif (x1.t == ('cautious')) & (x2.t == ('cautious')):
            subgame = 3


        # z1_dec = x1.z - round(x1.z)
        # z2_dec = x2.z - round(x2.z)

        z1_dec = x1.z
        z2_dec = x2.z

        row, col = map(lambda x: 0 if x < 0 else 1, [z1_dec, z2_dec])

        return subgame, row, col
