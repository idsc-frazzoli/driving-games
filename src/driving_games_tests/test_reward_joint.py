from decimal import Decimal as D
from typing import Hashable, Mapping

from matplotlib import pyplot as plt
from zuper_commons.text import remove_escapes
from zuper_typing import debug_print

from dg_commons import PlayerName, DgSampledSequence, fd
from dg_commons.sim.models.vehicle_ligths import NO_LIGHTS
from driving_games import (
    VehicleJointCost,
    VehicleSafetyDistCost,
    VehicleTrackState,
    VehicleTrackDynamics,
    VehicleActions,
    DrivingGameVisualization,
)
from driving_games.zoo_games import games_zoo, P4, mint_param_4p, P1, P2, P3
from driving_games_tests import logger
from games import JointTransition


def test_VehicleJointCost():
    test_1 = VehicleJointCost(VehicleSafetyDistCost(0))
    assert isinstance(test_1, Hashable)


def generate_transition(x: VehicleTrackState, dyn: VehicleTrackDynamics, u: VehicleActions, dt) -> VehicleTrackState:
    return dyn.successor(x, u, dt=dt)


def test_1():
    game = games_zoo["multilane_int_4p_sets"]().game

    dt = D("2")
    x1: VehicleTrackState = VehicleTrackState(x=D(14.0), v=D(1.0), wait=D(0), light=NO_LIGHTS, has_collided=False)
    x2: VehicleTrackState = VehicleTrackState(x=D(9.0), v=D(1.0), wait=D(0), light=NO_LIGHTS, has_collided=False)
    x3: VehicleTrackState = VehicleTrackState(x=D(14.0), v=D(1.0), wait=D(0), light=NO_LIGHTS, has_collided=False)
    x4: VehicleTrackState = VehicleTrackState(x=D(21.0), v=D(3.0), wait=D(0), light=NO_LIGHTS, has_collided=False)
    js0: Mapping[PlayerName, VehicleTrackState] = {P1: x1, P2: x2, P3: x3, P4: x4}
    dynamics = {p: game.players[p].dynamics for p in game.players}

    u1 = VehicleActions(acc=D(1), light=NO_LIGHTS)
    u2 = VehicleActions(acc=D(0), light=NO_LIGHTS)
    u3 = VehicleActions(acc=D(1), light=NO_LIGHTS)
    u4 = VehicleActions(acc=D(1), light=NO_LIGHTS)
    ju: Mapping[PlayerName, VehicleActions] = {P1: u1, P2: u2, P3: u3, P4: u4}
    js1 = {p: generate_transition(js0[p], dynamics[p], ju[p], dt) for p in game.players}
    txs: JointTransition = fd(
        {p: DgSampledSequence[VehicleTrackState](timestamps=(0, dt), values=(js0[p], js1[p])) for p in game.players}
    )

    joint_reward = game.joint_reward.joint_reward_incremental(txs)

    dg_vis = DrivingGameVisualization(
        mint_param_4p,
        geometries=game.joint_reward.geometries,
        dynamics=dynamics,
        plot_limits=mint_param_4p.plot_limits,  # param_3p.plot_limits
    )
    fig, ax = plt.subplots()
    ax.set_aspect(1)
    with dg_vis.plot_arena(plt, ax):
        for player_name in game.players:
            dg_vis.plot_player(player_name, js0[player_name], commands=None, t=0)
            dg_vis.plot_player(player_name, js1[player_name], commands=None, t=0)
    fig.set_tight_layout(True)
    fig.savefig("test_joint_reward.png", dpi=300)
    str_jr = remove_escapes(debug_print(joint_reward))
    logger.info(f"joint_reward={str_jr}")
