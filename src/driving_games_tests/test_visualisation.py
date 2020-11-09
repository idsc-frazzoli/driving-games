from typing import Mapping

from reprep import Report

from driving_games import p_asym, PlayerName, VehicleGeometry, VehicleState, NO_LIGHTS
from driving_games.visualization import AVAILABLE_CARS, DrivingGameVisualization
from decimal import Decimal as D
from matplotlib import pyplot as plt


def test_available_cars():
    # todo finish for testing better visualisations
    vehicles_params = p_asym
    L = vehicles_params.side + vehicles_params.road + vehicles_params.side
    P2 = PlayerName("W←")
    P1 = PlayerName("N↑")
    mass = D(1000)
    length = D(4.5)
    width = D(1.8)
    g1 = VehicleGeometry(mass=mass, width=width, length=length, color=(1, 0, 0))
    g2 = VehicleGeometry(mass=mass, width=width, length=length, color=(0, 0, 1))
    geometries = {P1: g1, P2: g2}
    start = vehicles_params.side + vehicles_params.road_lane_offset
    p1_ref = (D(start), D(0), D(+90))
    p2_ref = (D(L), D(start), D(-180))
    p1_x = VehicleState(
        ref=p1_ref,
        x=D(vehicles_params.first_progress),
        wait=D(0),
        v=D(0),
        light=NO_LIGHTS,
    )
    p2_x = VehicleState(
        ref=p2_ref,
        x=D(vehicles_params.second_progress),
        wait=D(0),
        v=D(0),
        light=NO_LIGHTS,
    )
    dg_vis = DrivingGameVisualization(p_asym, L, geometries=geometries, ds=vehicles_params.shared_resources_ds)
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.set_aspect(1)
    report: Report = Report(nid="test_visualisation")
    # report.to_html(join(dg, "r_animation.r_game"))
    vehicle_states: Mapping[PlayerName, VehicleState] = {P1: p1_x, P2: p2_x}
    with dg_vis.plot_arena(plt, ax):
        for player_name in [P2, P1]:
            dg_vis.plot_player(
                player_name,
                vehicle_states[player_name],
                commands=None
            )
    fig.show()
