from dataclasses import replace

from matplotlib import pyplot as plt

from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from driving_games import VehicleTrackState
from driving_games.visualization import DrivingGameVisualization
from driving_games.zoo_games import *


def test_available_cars():
    # testing some basic visualisation. Might be useful also to find interesting initial conditions
    vehicles_params: DgSimpleParams = param_3p
    g1 = VehicleGeometry.default_car(color=(1, 0, 0))
    g2 = VehicleGeometry.default_car(color=(0, 0, 1))
    g3 = VehicleGeometry.default_car(color=(0, 1, 1))
    geometries = {P1: g1, P2: g2, P3: g3}
    p1_x = VehicleTrackState(
        x=D(vehicles_params.progress[P1][0]),
        wait=D(0),
        v=D(0),
        light=NO_LIGHTS,
        has_collided=False,
    )
    p2_x = replace(p1_x, x=D(vehicles_params.progress[P2][0]))
    p3_x = replace(p1_x, x=D(vehicles_params.progress[P3][0]))

    dg_vis = DrivingGameVisualization(
        param_3p, geometries=geometries, ds=vehicles_params.shared_resources_ds, plot_limits=param_3p.plot_limits
    )
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.set_aspect(1)
    vehicle_states: Mapping[PlayerName, VehicleTrackState] = {P1: p1_x, P2: p2_x, P3: p3_x}
    with dg_vis.plot_arena(plt, ax):
        for player_name in [P2, P1, P3]:
            dg_vis.plot_player(player_name, vehicle_states[player_name], commands=None, t=0)
    fig.savefig("test.png")
