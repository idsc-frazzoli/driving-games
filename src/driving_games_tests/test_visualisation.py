from decimal import Decimal as D
from typing import Mapping

from matplotlib import pyplot as plt
from reprep import Report

from dg_commons import PlayerName
from dg_commons.sim.models.vehicle_ligths import NO_LIGHTS
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from driving_games import VehicleTrackState
from driving_games.dg_def import DgSimpleParams
from driving_games.visualization import DrivingGameVisualization
from driving_games.zoo import p_asym, P2, P1


def test_available_cars():
    # todo finish for testing better visualisations
    vehicles_params: DgSimpleParams = p_asym
    g1 = VehicleGeometry.default_car(color=(1, 0, 0))
    g2 = VehicleGeometry.default_car(color=(0, 0, 1))
    geometries = {P1: g1, P2: g2}
    p1_x = VehicleTrackState(
        x=D(vehicles_params.progress[P1][0]),
        wait=D(0),
        v=D(0),
        light=NO_LIGHTS,
    )
    p2_x = VehicleTrackState(
        x=D(vehicles_params.progress[P2][0]),
        wait=D(0),
        v=D(0),
        light=NO_LIGHTS,
    )
    dg_vis = DrivingGameVisualization(
        p_asym, geometries=geometries, ds=vehicles_params.shared_resources_ds, plot_limits=p_asym.plot_limits
    )
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.set_aspect(1)
    vehicle_states: Mapping[PlayerName, VehicleTrackState] = {P1: p1_x, P2: p2_x}
    with dg_vis.plot_arena(plt, ax):
        for player_name in [P2, P1]:
            dg_vis.plot_player(player_name, vehicle_states[player_name], commands=None, t=0)
    fig.savefig("test.png")
