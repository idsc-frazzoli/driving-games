from matplotlib import pyplot as plt
from shapely.strtree import STRtree

from dg_commons.maps.shapely_viz import ShapelyViz
from driving_games.dg_def import DgSimpleParams
from driving_games.resources_occupancy import ResourcesOccupancy
from driving_games.zoo_games import games_zoo
from driving_games.zoo_solvers import solvers_zoo


def test_resources():
    # complex_int_6p_sets
    game = games_zoo["4way_int_2p_sets"].game
    # solver = solvers_zoo["solver-2-pure-security_mNE-fact-noextra"]
    dg_params: DgSimpleParams = game.game_visualization.params
    res_occupancy = ResourcesOccupancy(lanelet_network=dg_params.scenario.lanelet_network, cell_resolution=5)

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
