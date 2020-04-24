import itertools
from dataclasses import dataclass
from typing import AbstractSet, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from typing_extensions import Literal

from driving_games.game_def import (
    Dynamics,
    Game,
    GamePlayer,
    JointRewardStructure,
    Observations,
    PersonalRewardStructure, PlayerName, Poset,
)
from geometry import SE2, SE2_from_xytheta, xytheta_from_SE2
from . import logger

Lights = Literal["none", "headlights", "turn_left", "turn_right"]
# noinspection PyTypeChecker
LightsValue: Sequence[Lights] = ["none", "headlights", "turn_left", "turn_right"]

SE2_disc = Tuple[int, int, int]  # in degrees


@dataclass(frozen=True)
class VehicleState:
    # reference frame from where the vehicle started
    ref: SE2_disc
    x: int
    v: int
    # how long at speed = 0 (we want to bound)
    wait: int
    light: Lights


@dataclass(frozen=True)
class VehicleActions:
    accel: int
    light: Lights


class VehicleDynamics(Dynamics[VehicleState, VehicleActions]):
    max_speed: int
    available_accels: AbstractSet[int]
    max_wait: int

    def __init__(
        self, max_speed: int, available_accels: AbstractSet[int], max_wait: int, ref, max_path
    ):
        self.max_speed = max_speed
        self.available_accels = available_accels
        self.max_wait = max_wait
        self.ref =ref
        self.max_path =max_path

    def all_actions(self) -> AbstractSet[VehicleActions]:
        res = set()
        for light, accel in itertools.product(LightsValue, self.available_accels):
            res.add(VehicleActions(accel=accel, light=light))
        return res

    def successors(
        self, x: VehicleState
    ) -> Mapping[VehicleActions, AbstractSet[VehicleState]]:
        """ For each state, returns a dictionary U -> Possible Xs """
        # only allow accellerations that make the speed non-negative
        accels = [_ for _ in self.available_accels if _ + x.v >= 0]
        # if the speed is 0 make sure we cannot wait forever
        if x.wait >= self.max_wait:
            assert x.v == 0, x
            accels.remove(0)

        possible = {}
        for light, accel in itertools.product(LightsValue, self.available_accels):
            u = VehicleActions(accel=accel, light=light)
            x2 = self.successor(x, u)
            possible[u] = {x2}

        return possible

    def successor(self, x: VehicleState, u: VehicleActions):
        v2 = x.v + u.accel
        assert v2 >= 0
        x2 = x.x + (x.v + u.accel)
        if v2 == 0:
            wait2 = x.wait + 1
        else:
            wait2 = 0
        return VehicleState(ref=x.ref, x=x2, v=v2, wait=wait2, light=u.light)

    def all_states(self) -> AbstractSet[VehicleState]:
        res = set()
        for l in LightsValue:
            for x in range(self.max_path):
                for wait in range(self.max_wait):
                    speed = 0
                    s = VehicleState(ref=self.ref, x=x, v=speed, wait=wait, light=l)
                    res.add(s)
                for speed in range(self.max_speed):
                    wait = 0
                    s = VehicleState(ref=self.ref, x=x, v=speed, wait=wait, light=l)
                    res.add(s)
        return res


@dataclass
class NotSeen:
    pass


@dataclass
class Seen:
    ref: SE2_disc
    x: Optional[int]
    v: Optional[int]
    # if not None, we could also see the light value
    light: Optional[Lights]


@dataclass
class VehicleObservation:
    others: Mapping[PlayerName, Union[Seen, NotSeen]]


class VehicleDirectObservations(Observations[VehicleState, VehicleObservation]):
    possible_states: Mapping[PlayerName, AbstractSet[VehicleState]]
    my_possible_states: AbstractSet[VehicleState]

    def __init__(
        self,
        my_possible_states: AbstractSet[VehicleState],
        possible_states: Mapping[PlayerName, AbstractSet[VehicleState]],
    ):
        self.possible_states = possible_states
        self.my_possible_states = my_possible_states

    def all_observations(self) -> AbstractSet[VehicleObservation]:
        """ Returns all possible observations. """
        assert len(self.possible_states) == 1
        all_of_them = set()
        for me in self.my_possible_states:
            for k, ks_possible_states in self.possible_states.items():
                for ks_possible_state in ks_possible_states:
                    others = {k: ks_possible_state}
                    possible_ys: AbstractSet[
                        VehicleObservation
                    ] = self.get_observations(me, others)
                    for poss_obs in possible_ys:
                        all_of_them.add(poss_obs)
        return all_of_them

    def get_observations(
        self, me: VehicleState, others: Mapping[PlayerName, VehicleState]
    ) -> AbstractSet[VehicleObservation]:
        # ''' For each state, get all possible observations '''
        others = {}
        for k, v in others.items():
            others[k] = Seen(ref=v.ref, x=v.x, v=v.v, light=None)
        return {VehicleObservation(others)}


X_ = VehicleState
U_ = VehicleActions
Y_ = VehicleObservation
RP_ = int
RJ_ = bool


class VehiclePersonalRewardStructureTime(PersonalRewardStructure):
    max_path: int

    def __init__(self, max_path: int):
        self.max_path = max_path

    def personal_reward_incremental(self, x: VehicleState, u: VehicleActions) -> int:
        return 1

    def personal_reward_reduce(self, r1: int, r2: int) -> int:
        return r1 + r2

    def is_personal_final_state(self, x: VehicleState) -> bool:
        return x.x > self.max_path


class VehiclePreferencesCollTime(Poset[Tuple[Optional[bool], int]]):

    def leq(self, a: Tuple[Optional[bool], int], b: Tuple[Optional[bool], int]) -> bool:
        collision_a, time_a = a
        collision_b, time_b = b

        col = {
            (None, True): True,
            # (None, False):
            (True, None): False,
            (False, True): True,
            (True, False): False
        }
        if (collision_a, collision_b) in col:
            return col[(collision_a, collision_b)]

        return a <= b


@dataclass
class TwoVehicleSimpleParams:
    side: int
    road: int
    road_lane_offset: int
    max_speed: int
    max_wait: int
    available_accels: AbstractSet[int]
    collision_threshold: float


def get_game1() -> Game:
    p = TwoVehicleSimpleParams(side=8, road=6, road_lane_offset=4,
                               max_speed=5, max_wait=3,
                               available_accels={-2, 0, +1},
                               collision_threshold=3.0
                               )
    return get_two_vehicle_game(p)


def SE2_from_VehicleState(s: VehicleState):
    p = SE2_from_xytheta([s.x, 0, 0])
    ref = SE2_from_xytheta([s.ref[0], s.ref[1], np.deg2rad(s.ref[2])])
    return SE2.multiply(ref, p)


def pose_diff(a, b):
    S = SE2
    return S.multiply(S.inverse(a), b)


class VehicleJointReward(JointRewardStructure):
    def __init__(self, collision_threshold: float):
        self.collision_threshold = collision_threshold

    def is_joint_final_state(self, xs: Mapping[PlayerName, VehicleState]) -> AbstractSet[PlayerName]:
        if len(xs) != 2:
            raise NotImplementedError(len(xs))
        p1, p2 = list(xs.values())
        c1 = SE2_from_VehicleState(p1)
        c2 = SE2_from_VehicleState(p2)
        d = pose_diff(c1, c2)
        x, y, _ = xytheta_from_SE2(d)
        dist = np.hypot(x, y)
        if dist < self.collision_threshold:
            return set(xs)
        else:
            return set()

    def joint_reward(self, xs: Mapping[PlayerName, VehicleState]) -> Mapping[PlayerName, bool]:
        players = self.is_joint_final_state(xs)
        if not players:
            raise Exception()
        res = {}
        for p in players:
            res[p] = True
        return res


def get_two_vehicle_game(params: TwoVehicleSimpleParams) -> Game:
    L = params.side + params.road + params.side
    start = params.side + params.road_lane_offset
    # p1_ref = SE2_from_xytheta([start, 0, np.pi / 2])
    p1_ref = (start, 0, +90)
    # p2_ref = SE2_from_xytheta([L, start, -np.pi])
    p2_ref = (start, 0, -180)
    max_path = L
    max_speed = params.max_speed
    max_wait = params.max_wait
    available_accels = params.available_accels

    P1 = PlayerName("p1")
    P2 = PlayerName("p2")
    p1_initial = {VehicleState(p1_ref, 0, 0, 0, "none")}
    p2_initial = {VehicleState(p2_ref, 0, 0, 0, "none")}
    p1_dynamics  = VehicleDynamics(
        max_speed=max_speed, max_wait=max_wait, available_accels=available_accels,
        max_path=max_path, ref=p1_ref
    )
    p2_dynamics = VehicleDynamics(
        max_speed=max_speed, max_wait=max_wait, available_accels=available_accels,
        max_path=max_path, ref=p2_ref
    )
    p1_possible_states = p1_dynamics.all_states()
    p2_possible_states = p2_dynamics.all_states()
    logger.info('npossiblestates', p1=len(p1_possible_states),
                p2=len(p2_possible_states))
    p1_observations = VehicleDirectObservations(
        my_possible_states=p1_possible_states, possible_states={P2: p2_possible_states}
    )
    p2_observations = VehicleDirectObservations(
        my_possible_states=p2_possible_states, possible_states={P1: p1_possible_states}
    )
    p1_personal_reward_structure = p2_personal_reward_structure = VehiclePersonalRewardStructureTime(max_path)
    p1_preferences = p2_preferences = VehiclePreferencesCollTime()
    p1 = GamePlayer(
        initial=p1_initial,
        dynamics=p1_dynamics,
        observations=p1_observations,
        personal_reward_structure=p1_personal_reward_structure,
        preferences=p1_preferences,
    )
    p2 = GamePlayer(
        initial=p2_initial,
        dynamics=p2_dynamics,
        observations=p2_observations,
        personal_reward_structure=p2_personal_reward_structure,
        preferences=p2_preferences,
    )

    players: Mapping[PlayerName, GamePlayer[X_, U_, Y_, RP_, RJ_]] = {
        P1: p1,
        P2: p2,
    }
    joint_reward: JointRewardStructure[X_, U_, RJ_] = VehicleJointReward(collision_threshold=params.collision_threshold)

    game: Game[X_, U_, Y_, RP_, RJ_] = Game(players, joint_reward)
    return game
