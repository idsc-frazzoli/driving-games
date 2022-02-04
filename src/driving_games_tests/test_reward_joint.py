from dataclasses import replace
from decimal import Decimal as D
from math import pi
from typing import Mapping

from dg_commons import PlayerName, SE2Transform, DgSampledSequence
from dg_commons.maps import DgLanelet, LaneCtrPoint
from dg_commons.sim.models.vehicle_ligths import NO_LIGHTS
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from driving_games import VehicleTrackState
from driving_games.reward_joint import VehicleJointReward
from games.game_def import JointTransition

P1 = PlayerName("p1")
P2 = PlayerName("p2")
P3 = PlayerName("p3")


def test_1():
    geo: Mapping[PlayerName, VehicleGeometry] = {
        P1: VehicleGeometry.default_car(),
        P2: VehicleGeometry.default_car(),
        P3: VehicleGeometry.default_bicycle(),
    }

    l1 = DgLanelet(
        [
            LaneCtrPoint(SE2Transform(p=[0, -2], theta=pi / 2), r=2),
            LaneCtrPoint(SE2Transform(p=[0, 2], theta=pi / 2), r=2),
        ]
    )
    l2 = DgLanelet(
        [LaneCtrPoint(SE2Transform(p=[-2, 0], theta=0), r=2), LaneCtrPoint(SE2Transform(p=[2, 0], theta=0), r=2)]
    )
    l3 = DgLanelet(
        [
            LaneCtrPoint(SE2Transform(p=[2, 2], theta=-pi * 3 / 2), r=2),
            LaneCtrPoint(SE2Transform(p=[-2, -2], theta=-pi * 3 / 2), r=2),
        ]
    )

    ref_lanes: Mapping[PlayerName, DgLanelet] = {P1: l1, P2: l2, P3: l3}
    jr = VehicleJointReward(geometries=geo, ref_lanes=ref_lanes, col_check_dt=0.2, min_safety_distance=3)

    # todo create some of the joint transitions
    dt = 3
    x0 = VehicleTrackState(x=D(0), v=D(1), wait=D(0), light=NO_LIGHTS)
    txs: Mapping[PlayerName, JointTransition] = {
        P1: DgSampledSequence[VehicleTrackState](timestamps=(0, dt), values=(x0, replace(x0, x=D(4), v=D(0)))),
        P2: DgSampledSequence[VehicleTrackState](timestamps=(0, dt), values=(x0, replace(x0, x=D(4), v=D(0)))),
        P3: DgSampledSequence[VehicleTrackState](timestamps=(0, dt), values=(x0, replace(x0, x=D(4), v=D(0)))),
    }

    res = jr.joint_reward_incremental(txs)
