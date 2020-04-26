import itertools
from dataclasses import dataclass
from decimal import Decimal as D
from functools import lru_cache
from typing import cast, FrozenSet as ASet, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from frozendict import frozendict
from typing_extensions import Literal

from geometry import SE2, SE2_from_xytheta, xytheta_from_SE2
from zuper_commons.types import check_isinstance, ZException, ZValueError
from zuper_typing import debug_print
from . import logger
from .access import get_accessible_states
from .game_def import (
    Combined,
    Dynamics,
    Game,
    GamePlayer,
    JointRewardStructure,
    Observations,
    PersonalRewardStructure,
    PlayerName,
)
from .poset import (
    COMP_OUTCOMES,
    ComparisonOutcome,
    FIRST_PREFERRED,
    INDIFFERENT,
    Preference,
    SECOND_PREFERRED,
    SmallerPreferred,
)
from .poset_lexi import LexicographicPreference

Lights = Literal["none", "headlights", "turn_left", "turn_right"]
# noinspection PyTypeChecker
LightsValue: Sequence[Lights] = ["none", "headlights", "turn_left", "turn_right"]
NO_LIGHTS = cast(Lights, "none")
SE2_disc = Tuple[D, D, D]  # in degrees


class InvalidAction(ZException):
    pass


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class VehicleState:
    # reference frame from where the vehicle started
    ref: SE2_disc
    x: D
    v: D
    # how long at speed = 0 (we want to bound)
    wait: D
    light: Lights


@dataclass(frozen=True)
class VehicleActions:
    accel: D
    light: Lights = "none"


class VehicleDynamics(Dynamics[VehicleState, VehicleActions]):
    max_speed: D
    min_speed: D
    max_path: D
    available_accels: ASet[D]
    max_wait: D
    lights_commands: ASet[Lights]

    def __init__(
        self,
        max_speed: D,
        min_speed: D,
        available_accels: ASet[D],
        max_wait: D,
        ref: SE2_disc,
        max_path: D,
        lights_commands: ASet[Lights],
    ):
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.available_accels = available_accels
        self.max_wait = max_wait
        self.ref = ref
        self.max_path = max_path
        self.lights_commands = lights_commands

    @lru_cache(None)
    def all_actions(self) -> ASet[VehicleActions]:
        res = set()
        for light, accel in itertools.product(LightsValue, self.available_accels):
            res.add(VehicleActions(accel=accel, light=light))
        return frozenset(res)

    @lru_cache(None)
    def successors(self, x: VehicleState, dt: D) -> Mapping[VehicleActions, ASet[VehicleState]]:
        """ For each state, returns a dictionary U -> Possible Xs """
        # only allow accellerations that make the speed non-negative
        accels = [_ for _ in self.available_accels if _ + x.v >= 0]
        # if the speed is 0 make sure we cannot wait forever
        if x.wait >= self.max_wait:
            assert x.v == 0, x
            accels.remove(D(0))

        possible = {}
        for light, accel in itertools.product(self.lights_commands, self.available_accels):
            u = VehicleActions(accel=accel, light=light)
            try:
                x2 = self.successor(x, u, dt)
            except InvalidAction:
                pass
            else:
                possible[u] = frozenset({x2})

        return frozendict(possible)

    @lru_cache(None)
    def successor(self, x: VehicleState, u: VehicleActions, dt: D):
        v2 = x.v + u.accel * dt
        if v2 < 0:
            v2 = 0
            # msg = 'Invalid action gives negative vel'
            # raise InvalidAction(msg, x=x, u=u)
        # if v2 < self.min_speed:
        #     v2 = self.min_speed
        if v2 > self.max_speed:
            v2 = self.max_speed
        if not (self.min_speed <= v2 <= self.max_speed):
            msg = "Invalid action gives speed too fast"
            raise InvalidAction(msg, x=x, u=u, v2=v2, max_speed=self.max_speed)
        assert v2 >= 0
        x2 = x.x + (x.v + u.accel * dt) * dt
        if x2 > self.max_path:
            msg = "Invalid action gives out of bound"
            raise InvalidAction(msg, x=x, u=u, v2=v2, max_speed=self.max_speed)
        # if wait2 > self.max_wait:
        #     msg = f'Invalid action gives wait of {wait2}'
        #     raise InvalidAction(msg, x=x, u=u)

        if v2 == 0:
            wait2 = x.wait + dt
            if wait2 > self.max_wait:
                msg = f"Invalid action gives wait of {wait2}"
                raise InvalidAction(msg, x=x, u=u)
        else:
            wait2 = 0
        return VehicleState(ref=x.ref, x=x2, v=v2, wait=wait2, light=u.light)

    # @lru_cache(None)
    # def all_states(self) -> ASet[VehicleState]:
    #     res = set()
    #     for l in self.lights_commands:
    #         for x in range(self.max_path + 1):
    #             for wait in range(self.max_wait + 1):
    #                 speed = 0
    #                 s = VehicleState(ref=self.ref, x=x, v=speed, wait=wait, light=l)
    #                 res.add(s)
    #             for speed in range(self.max_speed + 1):
    #                 wait = 0
    #                 s = VehicleState(ref=self.ref, x=x, v=speed, wait=wait, light=l)
    #                 res.add(s)
    #     for x in res:
    #         self.assert_valid_state(x)
    #     return res

    @lru_cache(None)
    def assert_valid_state(self, s: VehicleState):
        if s.wait and s.v:
            raise ZValueError(s=s)

        if not (0 <= s.x <= self.max_path):
            raise ZValueError(s=s)
        if not (0 <= s.v <= self.max_speed):
            raise ZValueError(s=s)
        if not (0 <= s.wait <= self.max_wait):
            raise ZValueError(s=s)


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
    possible_states: Mapping[PlayerName, ASet[VehicleState]]
    my_possible_states: ASet[VehicleState]

    def __init__(
        self,
        my_possible_states: ASet[VehicleState],
        possible_states: Mapping[PlayerName, ASet[VehicleState]],
    ):
        self.possible_states = possible_states
        self.my_possible_states = my_possible_states

    @lru_cache(None)
    def all_observations(self) -> ASet[VehicleObservation]:
        """ Returns all possible observations. """
        assert len(self.possible_states) == 1
        all_of_them = set()
        for me in self.my_possible_states:
            for k, ks_possible_states in self.possible_states.items():
                for ks_possible_state in ks_possible_states:
                    others = {k: ks_possible_state}
                    possible_ys: ASet[VehicleObservation] = self.get_observations(me, others)
                    for poss_obs in possible_ys:
                        all_of_them.add(poss_obs)
        return frozenset(all_of_them)

    @lru_cache(None)
    def get_observations(
        self, me: VehicleState, others: Mapping[PlayerName, VehicleState]
    ) -> ASet[VehicleObservation]:
        # ''' For each state, get all possible observations '''
        others = {}
        for k, v in others.items():
            others[k] = Seen(ref=v.ref, x=v.x, v=v.v, light=None)
        return frozenset({VehicleObservation(others)})


X_ = VehicleState
U_ = VehicleActions
Y_ = VehicleObservation
RP_ = D


@dataclass(frozen=True, unsafe_hash=True, order=True)
class CollisionCost:
    v: D


RJ_ = CollisionCost


class VehiclePersonalRewardStructureTime(PersonalRewardStructure):
    max_path: D

    def __init__(self, max_path: D):
        self.max_path = max_path

    def personal_reward_incremental(self, x: VehicleState, u: VehicleActions, dt: D) -> D:
        return dt

    def personal_reward_reduce(self, r1: D, r2: D) -> D:
        return r1 + r2

    def personal_final_reward(self, x: VehicleState) -> D:
        # assert self.is_personal_final_state(x)
        remaining = (self.max_path - x.x) / x.v
        return remaining

    def is_personal_final_state(self, x: VehicleState) -> bool:
        # return x.x > self.max_path

        return x.x + x.v > self.max_path


class CollisionPreference(Preference[Optional[CollisionCost]]):
    def __init__(self):
        self.p = SmallerPreferred()

    def get_type(self):
        return Optional[CollisionCost]

    def compare(self, a: Optional[CollisionCost], b: Optional[CollisionCost]) -> ComparisonOutcome:
        if a is None and b is None:
            return INDIFFERENT
        if a is None:
            return FIRST_PREFERRED
        if b is None:
            return SECOND_PREFERRED
        res = self.p.compare(a.v, b.v)
        assert res in COMP_OUTCOMES, (res, self.p)
        return res

    def __repr__(self):
        d = {
            "T": self.get_type(),
            "p": self.p,
        }
        return "CollisionPreference:\n " + debug_print(d)


class VehiclePreferencesCollTime(Preference[Combined[CollisionCost, D]]):
    def __init__(self):
        collision = CollisionPreference()
        time = SmallerPreferred()
        self.lexi = LexicographicPreference((collision, time))

    def get_type(self):
        return Combined[CollisionCost, D]

    def __repr__(self):
        d = {"P": self.get_type(), "lexi": self.lexi}
        return "VehiclePreferencesCollTime: " + debug_print(d)

    def compare(
        self, a: Combined[CollisionCost, D], b: Combined[CollisionCost, D]
    ) -> ComparisonOutcome:
        check_isinstance(a, Combined)
        check_isinstance(b, Combined)
        ct_a = (a.joint, a.personal)
        ct_b = (b.joint, b.personal)
        res = self.lexi.compare(ct_a, ct_b)
        assert res in COMP_OUTCOMES, (res, self.lexi)
        return res


@dataclass
class TwoVehicleSimpleParams:
    side: D
    road: D
    road_lane_offset: D
    max_speed: D
    min_speed: D
    max_wait: D
    available_accels: ASet[D]
    collision_threshold: float
    light_actions: ASet[Lights]
    dt: D


def get_game1() -> Game:
    p = TwoVehicleSimpleParams(
        side=D(8),
        road=D(6),
        road_lane_offset=D(4),
        max_speed=D(5),
        min_speed=D(1),
        max_wait=D(1),
        # available_accels={D(-2), D(0), D(+1)},
        available_accels=frozenset({D(-2), D(-1), D(0), D(+1)}),
        collision_threshold=3.0,
        light_actions=frozenset({NO_LIGHTS}),
        dt=D(1),
    )
    return get_two_vehicle_game(p)


def SE2_from_VehicleState(s: VehicleState):
    p = SE2_from_xytheta([float(s.x), 0, 0])
    ref = SE2_from_xytheta([float(s.ref[0]), float(s.ref[1]), np.deg2rad(float(s.ref[2]))])
    return SE2.multiply(ref, p)


def pose_diff(a, b):
    S = SE2
    return S.multiply(S.inverse(a), b)


def sample_from_traj(s: VehicleState, dt: D, n: int) -> Tuple[Tuple[float, float], ...]:
    ref = SE2_from_xytheta([float(s.ref[0]), float(s.ref[1]), np.deg2rad(float(s.ref[2]))])
    res = []
    for i in range(-n, +n + 1):
        x2 = s.x + s.v * D(i) * dt
        p = SE2_from_xytheta([float(x2), 0, 0])
        p2 = SE2.multiply(ref, p)
        x1, y1, _ = xytheta_from_SE2(p2)
        res.append((x1, y1))
    return tuple(res)


class VehicleJointReward(JointRewardStructure[VehicleState, VehicleActions, CollisionCost]):
    def __init__(self, collision_threshold: float):
        self.collision_threshold = collision_threshold

    # @lru_cache(None)
    def is_joint_final_state(self, xs: Mapping[PlayerName, VehicleState]) -> ASet[PlayerName]:
        if len(xs) == 1:
            return frozenset()
        if len(xs) != 2:
            raise NotImplementedError(len(xs))
        s1, s2 = list(xs.values())
        mind = 1000
        dt = D(0.5)
        n = 2
        samples1 = sample_from_traj(s1, dt=dt, n=n)
        samples2 = sample_from_traj(s2, dt=dt, n=n)
        for (x1, y1), (x2, y2) in itertools.product(samples1, samples2):
            dist = np.hypot(x1 - x2, y1 - y2)
            mind = min(mind, dist)
        # d = pose_diff(c1, c2)
        # x, y, _ = xytheta_from_SE2(d)
        # dist = np.hypot(x, y)
        # logger.info(c1=xytheta_from_SE2(c1), c2=xytheta_from_SE2(c2), dist=dist)
        if mind < self.collision_threshold:
            return frozenset(xs)
        else:
            return frozenset()

    def joint_reward(
        self, xs: Mapping[PlayerName, VehicleState]
    ) -> Mapping[PlayerName, CollisionCost]:
        players = self.is_joint_final_state(xs)
        if not players:
            raise Exception()
        res = {}
        for p in players:
            res[p] = CollisionCost(xs[p].v)
        return res


def get_two_vehicle_game(params: TwoVehicleSimpleParams) -> Game:
    L = params.side + params.road + params.side
    start = params.side + params.road_lane_offset
    max_path = L - 1
    # p1_ref = SE2_from_xytheta([start, 0, np.pi / 2])
    p1_ref = (D(start), D(0), D(+90))
    # p2_ref = SE2_from_xytheta([L, start, -np.pi])
    p2_ref = (D(L - 1), D(start), D(-180))
    max_speed = params.max_speed
    min_speed = params.min_speed
    max_wait = params.max_wait
    dt = params.dt
    available_accels = params.available_accels

    P1 = PlayerName("p1")
    P2 = PlayerName("p2")
    p1_initial = frozenset({VehicleState(ref=p1_ref, x=D(0), wait=D(0), v=min_speed, light="none")})
    p2_initial = frozenset({VehicleState(ref=p2_ref, x=D(0), wait=D(0), v=min_speed, light="none")})
    p1_dynamics = VehicleDynamics(
        max_speed=max_speed,
        max_wait=max_wait,
        available_accels=available_accels,
        max_path=max_path,
        ref=p1_ref,
        lights_commands=params.light_actions,
        min_speed=min_speed,
    )
    p2_dynamics = VehicleDynamics(
        min_speed=min_speed,
        max_speed=max_speed,
        max_wait=max_wait,
        available_accels=available_accels,
        max_path=max_path,
        ref=p2_ref,
        lights_commands=params.light_actions,
    )
    p1_personal_reward_structure = VehiclePersonalRewardStructureTime(max_path)
    p2_personal_reward_structure = VehiclePersonalRewardStructureTime(max_path)

    g1 = get_accessible_states(p1_initial, p1_personal_reward_structure, p1_dynamics, dt)
    p1_possible_states = frozenset(g1.nodes)
    g2 = get_accessible_states(p2_initial, p2_personal_reward_structure, p2_dynamics, dt)
    p2_possible_states = frozenset(g2.nodes)

    logger.info("npossiblestates", p1=len(p1_possible_states), p2=len(p2_possible_states))
    p1_observations = VehicleDirectObservations(p1_possible_states, {P2: p2_possible_states})
    p2_observations = VehicleDirectObservations(p2_possible_states, {P1: p1_possible_states})

    p1_preferences = VehiclePreferencesCollTime()
    p2_preferences = VehiclePreferencesCollTime()
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
    players: Mapping[PlayerName, GamePlayer[X_, U_, Y_, RP_, RJ_]]
    players = {P1: p1, P2: p2}
    joint_reward: JointRewardStructure[X_, U_, RJ_]
    joint_reward = VehicleJointReward(collision_threshold=params.collision_threshold)

    game: Game[X_, U_, Y_, RP_, RJ_] = Game(players, joint_reward)
    return game
