from decimal import Decimal as D
from typing import Tuple

import networkx as nx
from networkx import convert_node_labels_to_integers

from reprep import MIME_GRAPHML, Report

from . import logger
from .driving_example import VehicleState
from .game_def import GamePlayer, GamePreprocessed, PlayerName


def create_report_preprocessed(game_name: str, game_pre: GamePreprocessed) -> Report:
    r = Report(nid=game_name)
    for player_name, player in game_pre.game.players.items():
        r.add_child(report_player(game_pre, player_name, player))
        break
    r.add_child(report_game(game_pre))
    return r


def report_player(game_pre: GamePreprocessed, player_name: PlayerName, player: GamePlayer):
    G = game_pre.players_pre[player_name].player_graph
    r = Report(nid=player_name)

    with r.data_file(('player'), mime=MIME_GRAPHML) as fn:
        G2 = convert_node_labels_to_integers(G)
        for (n1, n2, d) in G2.edges(data=True):
            d.clear()
        nx.write_graphml(G2, fn)

    def color_node(n):
        is_final = G.nodes[n]['is_final']
        return 'blue' if is_final else 'green'

    def pos_node(n: VehicleState):
        w = -n.wait * D(0.2)
        return float(n.x), float(n.v + w)

    node_size = 20
    node_color = [color_node(_) for _ in G.nodes]
    pos = {_: pos_node(_) for _ in G.nodes}

    with r.plot('one') as plt:
        nx.draw(G, pos=pos, node_color=node_color, cmap=plt.cm.Blues,
                node_size=node_size)
        plt.xlabel('x')
        plt.ylabel('v')
    logger.info('layout')
    #
    # pos = graphviz_layout(G, prog='dot')
    # logger.info('drawing')
    # with r.plot('s') as plt:
    #     nx.draw(G, pos=pos, node_color=node_color, cmap=plt.cm.Blues,
    #             node_size=node_size)
    #     plt.xlabel('x')
    #     plt.ylabel('v')

    return r


#
# def igraph_from_nx(G: DiGraph):
#     nx.write_graphml(G, 'graph.graphml')  # Export NX graph to file
#
#     import igraph as ig
#     Gix = ig.read('graph.graphml', format="graphml")  # Create new IG graph from file
#     return Gix


def report_game(game_pre: GamePreprocessed) -> Report:
    G = game_pre.game_graph

    r = Report(nid='game')

    with r.data_file('game', mime=MIME_GRAPHML) as fn:
        logger.info(f'done writing {fn}')
        G2 = convert_node_labels_to_integers(G)
        for (n1, n2, d) in G2.edges(data=True):
            d.clear()
        nx.write_graphml(G2, fn)

    def color_node(n):
        is_initial = G.nodes[n]['is_initial']
        is_final1 = G.nodes[n]['is_final1']
        is_final2 = G.nodes[n]['is_final2']
        is_joint_final = G.nodes[n]['is_joint_final']
        in_game = G.nodes[n]['in_game']
        if is_initial:
            return 'red'
        if is_joint_final:
            return 'magenta'
        if in_game == 'AB':
            if is_final1 and is_final2:
                return 'black'
            return 'green'
        elif in_game == 'A':
            if is_final1:
                return 'teal'
            else:
                return 'blue'
        elif in_game == 'B':
            if is_final2:
                return 'orange'
            else:
                return 'yellow'

        return 'grey'

    caption = 'green: both playing, blue/yellow: only one (final:teal, magenta). Initial: red. Joint final: magenta'

    node_size = 3
    node_color = [color_node(_) for _ in G.nodes]
    # logger.info('layout')
    # pos = graphviz_layout(G, prog='dot')
    logger.info('drawing')

    def pos_node(n: Tuple[VehicleState, VehicleState]):
        x = G.nodes[n]['x']
        y = G.nodes[n]['y']
        return float(x), float(y)

    pos = {_: pos_node(_) for _ in G.nodes}

    with r.plot('s', caption=caption) as plt:
        nx.draw(G, pos=pos, node_color=node_color,
                cmap=plt.cm.Blues, arrows=False,
                edge_color=(0, 0, 0, 0.1),
                node_size=node_size)
        plt.xlabel('x')
        plt.ylabel('v')
    return r
