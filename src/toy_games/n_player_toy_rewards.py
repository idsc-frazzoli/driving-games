from typing import Any, FrozenSet, Mapping, Optional, Tuple, Type
import itertools
from zuper_commons.types import check_isinstance
from decimal import Decimal as D
from dataclasses import dataclass

from zuper_typing import debug_print

from games import JointRewardStructure, PlayerName, PersonalRewardStructure, Combined
from toy_games.n_player_toy_structures import ToyCarState, ToyCarCosts, ToyCarActions

from preferences import (
    LexicographicPreference,
    COMP_OUTCOMES,
    ComparisonOutcome,
    FIRST_PREFERRED,
    INDIFFERENT,
    Preference,
    SECOND_PREFERRED,
    SmallerPreferredTol,
)


class ToyCarPersonalRewardStructureCustom(PersonalRewardStructure[ToyCarState, ToyCarActions, ToyCarCosts]):
    max_path: int

    def __init__(self, max_path: int):
        self.max_path = max_path

    def personal_reward_identity(self) -> ToyCarCosts:
        return ToyCarCosts(D(0))

    def personal_reward_incremental(self, x: ToyCarState, u: ToyCarActions, dt: int) -> ToyCarCosts:
        check_isinstance(x, ToyCarState)
        check_isinstance(u, ToyCarActions)
        return ToyCarCosts(D(dt))

    def personal_reward_reduce(self, r1: ToyCarCosts, r2: ToyCarCosts) -> ToyCarCosts:
        return r1 + r2

    def personal_final_reward(self, x: ToyCarState) -> ToyCarCosts:
        check_isinstance(x, ToyCarState)
        remaining = self.max_path - x.along_lane

        return ToyCarCosts(D(remaining))

    def is_personal_final_state(self, x: ToyCarState) -> bool:
        check_isinstance(x, ToyCarState)
        return x.along_lane >= self.max_path

@dataclass(frozen=True)
class ToyCollision:
    active: bool


class ToyCarPreferences(Preference[Combined[ToyCollision, ToyCarCosts]]):
    def __init__(self, ignore_second = False):
        self.ignore_second = ignore_second
        self.collision = ToyCollisionPreference()
        self.time = SmallerPreferredTol(D(0))
        self.lexi = LexicographicPreference((self.collision, self.time))

    def get_type(self) -> Type[Combined[ToyCollision, ToyCarCosts]]:
        return Combined[ToyCollision, ToyCarCosts]

    def __repr__(self) -> str:
        d = {"P": self.get_type()}
        return "VehiclePreferencesCollTime: " + debug_print(d)

    def compare(
        self, a: Combined[ToyCollision, ToyCarCosts], b: Combined[ToyCollision, ToyCarCosts]
    ) -> ComparisonOutcome:
        if self.ignore_second:
            if a.joint is None and b.joint is None:
                return self.time.compare(a.personal.duration, b.personal.duration)
            else:
                return self.collision.compare(a.joint, b.joint)
        else:
            ct_a = (a.joint, a.personal.duration)
            ct_b = (b.joint, b.personal.duration)

            res = self.lexi.compare(ct_a, ct_b)
            assert res in COMP_OUTCOMES, (res, self.lexi)

            # logger.info(ct_a=ct_a, ct_b=ct_b, res=res)
            return res


class ToyCarJointReward(JointRewardStructure[ToyCarState, ToyCarActions, ToyCollision]):

    def is_joint_final_state(self, xs: Mapping[PlayerName, ToyCarState]) -> FrozenSet[PlayerName]:
        res = toy_collision_check(joint_state=xs)
        return frozenset(res)


    def joint_reward(self, xs: Mapping[PlayerName, ToyCarState]) -> Mapping[PlayerName, ToyCollision]:

        res = toy_collision_check(joint_state=xs)
        return res



def toy_collision_check(joint_state: Mapping[PlayerName, ToyCarState]) -> Mapping[PlayerName, ToyCollision]:
    collision_dict = {}

    players = list(joint_state)

    for p1, p2 in itertools.combinations(players, 2):

        if p1 in collision_dict or p2 in collision_dict:
            # Have already been in a collision or near a collision
            continue

        s1 = joint_state[p1]
        s2 = joint_state[p2]

        if not s1.point_in_map == s2.point_in_map:
            # no collision
            continue
        else:
            # collision
            c1 = ToyCollision(
                active=True
            )
            c2 = ToyCollision(
                active=True
            )

            two_player_col = {p1: c1, p2: c2}
            empty_col_dict = toy_stop_game_for_players_around(
                colliding_players=(p1, p2),
                joint_state=joint_state
            )

            collision_dict.update(empty_col_dict)
            collision_dict.update(two_player_col)

    return collision_dict


def toy_stop_game_for_players_around(
        colliding_players: Tuple[PlayerName, PlayerName],
        joint_state: Mapping[PlayerName, ToyCarState],
) -> Mapping[PlayerName, ToyCollision]:

    col_player1, col_player2 = colliding_players

    col_dict = {}
    empty_col = ToyCollision(
        active=False,
    )

    for player, state in joint_state.items():
        if player == col_player1 or player == col_player2:
            continue
        else:
            col_dict[player] = empty_col

    return col_dict


class ToyCollisionPreference(Preference[Optional[ToyCollision]]):

    def get_type(self) -> Type[Optional[ToyCollision]]:
        return Optional[ToyCollision]

    def compare(self, a: Optional[ToyCollision], b: Optional[ToyCollision]) -> ComparisonOutcome:
        if a is None and b is None:
            return INDIFFERENT
        if a is None and b is not None:
            return FIRST_PREFERRED
        if a is not None and b is None:
            return SECOND_PREFERRED
        assert a is not None
        assert b is not None
        if a.active and not b.active:
            return SECOND_PREFERRED
        if b.active and not a.active:
            return FIRST_PREFERRED

        assert False, "Should not happen"


    def __repr__(self) -> str:
        d = {
            "T": self.get_type(),
        }
        return "CollisionPreference:\n " + debug_print(d)
