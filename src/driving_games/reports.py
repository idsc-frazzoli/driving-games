from decimal import Decimal as D

import networkx as nx
from networkx import convert_node_labels_to_integers
from reprep import MIME_GRAPHML, Report

from games import GamePlayer, GamePreprocessed, PlayerName, RJ, RP, U, X, Y


def report_player(
    game_pre: GamePreprocessed[X, U, Y, RP, RJ],
    player_name: PlayerName,
    player: GamePlayer[X, U, Y, RP, RJ],
):
    pp = game_pre.players_pre[player_name]

    G = pp.player_graph
    r = Report(nid=player_name)

    with r.data_file(("player"), mime=MIME_GRAPHML) as fn:
        G2 = convert_node_labels_to_integers(G)
        for (n1, n2, d) in G2.edges(data=True):
            d.clear()
        nx.write_graphml(G2, fn)

    def color_node(n):
        is_final = G.nodes[n]["is_final"]
        return "blue" if is_final else "green"

    from driving_games.structures import VehicleState

    def pos_node(n: VehicleState):
        w = -n.wait * D(0.2)
        return float(n.x), float(n.v + w)

    node_size = 20
    node_color = [color_node(_) for _ in G.nodes]
    pos = {_: pos_node(_) for _ in G.nodes}

    with r.plot("one") as plt:
        nx.draw(G, pos=pos, node_color=node_color, cmap=plt.cm.Blues, node_size=node_size)
        plt.xlabel("x")
        plt.ylabel("v")
    return r
