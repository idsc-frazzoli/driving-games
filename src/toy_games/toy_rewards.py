from typing import Any, FrozenSet, Mapping, Sequence,  Tuple
import numpy as np
from zuper_commons.types import check_isinstance
from decimal import Decimal as D
from games import JointRewardStructure, PlayerName, PersonalRewardStructure
from toy_games.toy_structures import BirdState, BirdCosts, BirdActions
from preferences import SmallerPreferred


class BirdPersonalRewardStructureCustom(PersonalRewardStructure[BirdState, BirdActions, BirdCosts]):
    max_stages: int

    def __init__(self, max_stages: int):
        self.max_stages = max_stages

    def personal_reward_identity(self) -> BirdCosts:
        return BirdCosts(0)

    def personal_reward_incremental(self, x: BirdState, u: BirdActions, dt: D) -> BirdCosts:
        check_isinstance(x, BirdState)
        check_isinstance(u, BirdActions)
        return BirdCosts(0)

    def personal_reward_reduce(self, r1: BirdCosts, r2: BirdCosts) -> BirdCosts:
        return BirdCosts(r1.cost+r2.cost)

    def personal_final_reward(self, x: BirdState) -> BirdCosts:
        check_isinstance(x, BirdState)
        return BirdCosts(0)

    def is_personal_final_state(self, x: BirdState) -> bool:
        check_isinstance(x, BirdState)
        return x.stage == self.max_stages


class BirdPreferences(SmallerPreferred):
    ...


class BirdJointReward(JointRewardStructure[BirdState, BirdActions, Any]):
    max_stages: int
    leaves_payoffs: Mapping[BirdState, Mapping[PlayerName, BirdCosts]]
    row_player: PlayerName
    col_player: PlayerName
    mat_payoffs: Sequence[np.ndarray]

    def __init__(
            self,
            max_stages: int,
            leaves_payoffs: Sequence[np.ndarray],
            row_player: PlayerName,
            col_player: PlayerName,
    ):
        self.row_player = row_player
        self.col_player = col_player
        self.max_stages = max_stages
        assert len(leaves_payoffs) == 4, leaves_payoffs
        self.mat_payoffs = leaves_payoffs

    def is_joint_final_state(self, xs: Mapping[PlayerName, BirdState]) -> FrozenSet[PlayerName]:
        res = set()
        if len(xs.items()) > 1:
            for player, x in xs.items():
                if x.stage == self.max_stages:
                    res.add(player)
        return frozenset(res)

    def joint_reward(self, xs: Mapping[PlayerName, BirdState]) -> Mapping[PlayerName, Any]:
        """
        Each payoff matrix correspond to a specific game
        [-1,-1] -> leaves_payoffs[0]
        [-1,1] -> leaves_payoffs[1]
        [1,-1] -> leaves_payoffs[2]
        [1,1] -> leaves_payoffs[3]
        :param leaves_payoffs:
        :return:
        """
        res = {}
        x1, x2 = xs[self.row_player], xs[self.col_player]
        subgame, row, col = self.get_payoff_matrix_idx(x1, x2)
        payoff1, payoff2 = self.mat_payoffs[subgame][row, col]
        res.update({self.row_player: payoff1, self.col_player: payoff2})
        return res

    @staticmethod
    def get_payoff_matrix_idx(x1: BirdState, x2: BirdState) -> Tuple[int, int, int]:
        subgame: int
        row: int
        col: int

        thresh = 0
        if x1.z < thresh and x2.z < thresh:
            subgame = 0
        elif x1.z < thresh < x2.z:
            subgame = 1
        elif x1.z > thresh > x2.z:
            subgame = 2
        else:
            subgame = 3

        z1_dec = x1.z-round(x1.z)
        z2_dec = x2.z-round(x2.z)
        row, col = map(lambda x: 0 if x < 0 else 1, [z1_dec, z2_dec])

        return subgame, row, col
