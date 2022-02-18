from decimal import Decimal as D
from typing import Mapping

from matplotlib import pyplot as plt
from shapely.strtree import STRtree

from dg_commons import PlayerName
from dg_commons.maps.shapely_viz import ShapelyViz
from dg_commons.sim.models.vehicle_ligths import NO_LIGHTS
from driving_games import VehicleTrackState, VehicleTrackDynamics, DrivingGameVisualization
from driving_games.dg_def import DgSimpleParams
from driving_games.resources_occupancy import ResourcesOccupancy
from driving_games.zoo_games import games_zoo, P1, P2, P3, P4, mint_param_4p


def test_resources():
    game = games_zoo["4way_int_2p_sets"].game
    # solver = solvers_zoo["solver-2-pure-security_mNE-fact-noextra"]
    dg_params: DgSimpleParams = game.game_visualization.params
    res_occupancy = ResourcesOccupancy(lanelet_network=dg_params.scenario.lanelet_network, cell_resolution=D("1.5"))

    strtree: STRtree = res_occupancy.strtree
    viz = ShapelyViz()
    for poly in strtree._geoms:
        viz.add_shape(poly, edgecolor="k", facecolor="b", alpha=0.5)
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.autoscale()
    plt.savefig("resources_occupancy.png", dpi=300)

    # test queries
    # for pn in game.players:
    #     game.players[pn].dynamics.get_shared_resources()


def test_res_intersection():
    game = games_zoo["4way_int_3p_sets"].game
    p1_dyn: VehicleTrackDynamics = game.players[P1].dynamics
    p2_dyn: VehicleTrackDynamics = game.players[P2].dynamics
    dt = D("2")
    x1: VehicleTrackState = VehicleTrackState(x=D(140.0), v=D(2.0), wait=D(0), light=NO_LIGHTS, has_collided=False)
    p1_res = p1_dyn.get_shared_resources(x1, dt=dt)
    x2: VehicleTrackState = VehicleTrackState(x=D(185.0), v=D(2.0), wait=D(0), light=NO_LIGHTS, has_collided=False)
    p2_res = p2_dyn.get_shared_resources(x2, dt=dt)
    viz = ShapelyViz()
    # plot all resources:
    for poly in p1_dyn.resources_occupancy.strtree._geoms:
        viz.add_shape(poly, color="k", alpha=0.1)

    for poly_idx in p1_res:
        poly = p1_dyn.resources_occupancy.get_poly_from_idx(poly_idx)
        viz.add_shape(poly, color="yellow", alpha=0.5)

    for poly_idx in p2_res:
        poly = p2_dyn.resources_occupancy.get_poly_from_idx(poly_idx)
        viz.add_shape(poly, color="green", alpha=0.4)

    viz.ax.set_xlim(50, 90)
    viz.ax.set_ylim(-20, 25)
    viz.ax.set_aspect("equal")
    plt.savefig("test.png", dpi=300)

    print(p1_res)
    print(p2_res)
    print(p1_res & p2_res)


def test_res_intersection():
    game = games_zoo["multilane_int_4p_sets"].game

    dt = D("2")
    x1: VehicleTrackState = VehicleTrackState(x=D(14.0), v=D(1.0), wait=D(0), light=NO_LIGHTS, has_collided=False)
    x2: VehicleTrackState = VehicleTrackState(x=D(9.0), v=D(1.0), wait=D(0), light=NO_LIGHTS, has_collided=False)
    x3: VehicleTrackState = VehicleTrackState(x=D(14.0), v=D(1.0), wait=D(0), light=NO_LIGHTS, has_collided=False)
    x4: VehicleTrackState = VehicleTrackState(x=D(21.0), v=D(3.0), wait=D(0), light=NO_LIGHTS, has_collided=False)
    vehicle_states: Mapping[PlayerName, VehicleTrackState] = {P1: x1, P2: x2, P3: x3, P4: x4}

    dg_vis = DrivingGameVisualization(
        mint_param_4p,
        geometries=game.joint_reward.geometries,
        dynamics={p: game.players[p].dynamics for p in game.players},
        plot_limits=mint_param_4p.plot_limits,  # param_3p.plot_limits
    )
    fig, ax = plt.subplots()
    ax.set_aspect(1)
    with dg_vis.plot_arena(plt, ax):
        for player_name in game.players:
            dg_vis.plot_player(player_name, vehicle_states[player_name], commands=None, t=0)
    fig.set_tight_layout(True)
    fig.savefig("test2.png")

    # viz.ax.set_xlim(50, 90)
    # viz.ax.set_ylim(-20, 25)
    # viz.ax.set_aspect("equal")
    # plt.savefig("test.png", dpi=300)
    #
    # print(p1_res)
    # print(p2_res)
    # print(p1_res & p2_res)
