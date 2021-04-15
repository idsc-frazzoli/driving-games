from typing import Dict

from reprep import Report

from driving_games import PlayerName, NO_LIGHTS
from duckie_games.structures import DuckieState
from duckie_games.visualisation import DuckieGameVisualization
from duckie_games.zoo import two_player_4way
from world.utils import LaneSegmentHashable
from world.map_loading import load_driving_game_map
from decimal import Decimal as D
from matplotlib import pyplot as plt

from duckie_games.shared_resources import DrivingGameGridMap


def test_visualisation():
    """
    Visual test of the animation for the reports
    """
    duckie_game_params = two_player_4way
    map_name = duckie_game_params.map_name
    duckie_map = load_driving_game_map(map_name)
    shared_resources_ds = duckie_game_params.shared_resources_ds
    duckie_map_grid = DrivingGameGridMap.initializor(m=duckie_map, cell_size=shared_resources_ds)
    duckie_geometries = duckie_game_params.player_geometries

    duckie_states: Dict[PlayerName, DuckieState] = {}
    for duckie_name in duckie_game_params.player_names:

        ref = duckie_game_params.refs[duckie_name]

        lane = duckie_game_params.lanes[duckie_name]
        lane_hashable = LaneSegmentHashable.initializor(lane)

        ds = DuckieState(
            ref=ref,
            x=D(duckie_game_params.initial_progress[duckie_name]),
            lane=lane_hashable,
            wait=D(0),
            v=D(2),
            light=NO_LIGHTS
        )
        duckie_states[duckie_name] = ds

    dg_vis = DuckieGameVisualization(
        duckie_map=duckie_map_grid,
        map_name=map_name,
        geometries=duckie_geometries,
        ds=duckie_game_params.shared_resources_ds,
        dt=D(1)
    )
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.set_aspect(1)
    report: Report = Report(nid="test_visualisation")
    # report.to_html(join(dg, "r_animation.r_game"))

    with dg_vis.plot_arena(plt, ax):
        for player_name in duckie_game_params.player_names:
            dg_vis.plot_player(player_name, duckie_states[player_name], commands=None)
    #fig.show()
    plt.savefig("out/test_visualisation.png")

