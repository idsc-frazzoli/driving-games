from typing import Mapping, Dict

from reprep import Report

from driving_games import PlayerName, NO_LIGHTS
from duckie_games.structures import DuckieGeometry, DuckieState
from duckie_games.visualisation import DuckieGameVisualization
from duckie_games.zoo import two_player_duckie_game_parameters, two_player_duckie_game_parameters_stretched
from duckie_games.utils import LaneSegmentHashable
from decimal import Decimal as D
from matplotlib import pyplot as plt


def test_visualisation():
    duckie_game_params = two_player_duckie_game_parameters

    duckie_map = duckie_game_params.duckie_map
    map_name = duckie_game_params.map_name
    duckie_geometries = duckie_game_params.duckie_geometries

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
            v=duckie_game_params.min_speed[duckie_name],
            light=NO_LIGHTS
        )
        duckie_states[duckie_name] = ds

    dg_vis = DuckieGameVisualization(
        duckie_map=duckie_map, map_name=map_name, geometries=duckie_geometries, ds=duckie_game_params.shared_resources_ds
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

