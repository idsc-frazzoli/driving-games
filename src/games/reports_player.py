import networkx as nx
from networkx import convert_node_labels_to_integers
from reprep import MIME_GRAPHML, Report

from .game_def import GamePlayer, PlayerName, RJ, RP, U, X, Y
from .structures_solution import GamePreprocessed

__all__ = []


def report_player(
    game_pre: GamePreprocessed[X, U, Y, RP, RJ], player_name: PlayerName, player: GamePlayer[X, U, Y, RP, RJ],
):
    pp = game_pre.players_pre[player_name]
    viz = game_pre.game.game_visualization

    G = pp.player_graph
    r = Report(nid=player_name)

    with r.data_file(("player"), mime=MIME_GRAPHML) as fn:
        G2 = convert_node_labels_to_integers(G)
        for (n1, n2, d) in G2.edges(data=True):
            d.clear()
        nx.write_graphml(G2, fn)

    def color_node(n) -> str:
        is_final = G.nodes[n]["is_final"]
        return "blue" if is_final else "green"

    node_size = 20
    node_color = [color_node(_) for _ in G.nodes]
    pos = {_: viz.hint_graph_node_pos(_) for _ in G.nodes}

    with r.plot("one") as plt:
        nx.draw(G, pos=pos, node_color=node_color, cmap=plt.cm.Blues, node_size=node_size)
        plt.xlabel("x")
        plt.ylabel("v")
    return r
