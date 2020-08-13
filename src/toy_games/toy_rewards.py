from typing import Any, FrozenSet, Mapping, Sequence, Tuple, Type
import numpy as np
from zuper_commons.types import check_isinstance
from decimal import Decimal as D

from zuper_typing import debug_print

from games import JointRewardStructure, PlayerName, PersonalRewardStructure, Combined
from nash import BiMatGame
from toy_games.toy_structures import BirdState, BirdCosts, BirdActions
from preferences import SmallerPreferred, Preference, ComparisonOutcome


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
    def get_type(self) -> Type[Combined[D, D]]:
        return Combined[BirdCosts, BirdCosts]

    def __repr__(self) -> str:
        d = {"P": self.get_type()}
        return "VehiclePreferencesCollTime: "+debug_print(d)

    def compare(self, a: Combined[BirdCosts, BirdCosts],
                b: Combined[BirdCosts, BirdCosts]) -> ComparisonOutcome:
        a_ = a.personal+a.joint
        b_ = b.personal+b.joint
        return super().compare(D(a_.cost), D(b_.cost))


class BirdJointReward(JointRewardStructure[BirdState, BirdActions, Any]):
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
    ):
        self.row_player = row_player
        self.col_player = col_player
        self.max_stages = max_stages
        assert len(subgames) == 4, subgames
        self.mat_payoffs = [np.stack([g.A, g.B], axis=-1) for g in subgames]

    def is_joint_final_state(self, xs: Mapping[PlayerName, BirdState]) -> FrozenSet[PlayerName]:
        res = set()
        if len(xs.items()) > 1:
            for player, x in xs.items():
                if x.stage == self.max_stages:
                    res.add(player)
        return frozenset(res)

    def joint_reward(self, xs: Mapping[PlayerName, BirdState]) -> Mapping[PlayerName, BirdCosts]:
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
        payoff1, payoff2 = self.mat_payoffs[subgame][:, row, col]
        res.update({self.row_player: BirdCosts(payoff1), self.col_player: BirdCosts(payoff2)})
        return res

    @staticmethod
    def get_payoff_matrix_idx(x1: BirdState, x2: BirdState) -> Tuple[int, int, int]:
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
