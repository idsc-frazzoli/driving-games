from typing import Tuple

import networkx as nx
from networkx import convert_node_labels_to_integers
from reprep import MIME_GRAPHML, Report

from . import logger
from .game_def import Game, RJ, RP, U, X, Y
from .reports_player import report_player
from .structures_solution import GamePreprocessed

__all__ = ["create_report_preprocessed", "report_game_visualization", "report_game_joint_final"]


def create_report_preprocessed(game_name: str, game_pre: GamePreprocessed) -> Report:
    r = Report(nid=game_name)
    for player_name, player in game_pre.game.players.items():
        r.add_child(report_player(game_pre, player_name, player))
        # break  # only one
    r.add_child(report_game(game_pre))
    r.add_child(report_game_joint_final(game_pre))
    return r


def report_game_visualization(game: Game) -> Report:
    viz = game.game_visualization
    r = Report("vis")
    with r.plot("initial") as pylab:
        ax = pylab.gca()
        with viz.plot_arena(pylab, ax):
            for player_name, player in game.players.items():
                for x in player.initial:
                    viz.plot_player(player_name, state=x, commands=None)

    return r


def report_game_joint_final(game_pre: GamePreprocessed) -> Report:
    r = Report(nid="collisions")
    G = game_pre.game_graph

    final1 = [node for node in G if G.nodes[node]["is_final1"]]
    visualize_states(game_pre, r, "final1", final1[:5])
    final2 = [node for node in G if G.nodes[node]["is_final2"]]
    visualize_states(game_pre, r, "final2", final2[:5])
    joint_final = [node for node in G if G.nodes[node]["is_joint_final"]]
    visualize_states(game_pre, r, "joint_final", joint_final[:5])

    return r


def visualize_states(game_pre: GamePreprocessed[Pr, X, U, Y, RP, RJ], r: Report, name: str, nodes):
    viz = game_pre.game.game_visualization
    f = r.figure(name)
    for i, node in enumerate(nodes):
        with f.plot(f"f{i}") as pylab:
            ax = pylab.gca()
            with viz.plot_arena(pylab, ax):
                for player_name, player_state in node.items():
                    if player_state is not None:
                        viz.plot_player(player_name, state=player_state, commands=None)
    return f


def report_game(game_pre: GamePreprocessed) -> Report:
    G = game_pre.game_graph

    r = Report(nid="game")

    with r.data_file("game", mime=MIME_GRAPHML) as fn:
        logger.info(f"done writing {fn}")
        G2 = convert_node_labels_to_integers(G)
        for (n1, n2, d) in G2.edges(data=True):
            d.clear()
        nx.write_graphml(G2, fn)

    def color_node(n):
        is_initial = G.nodes[n]["is_initial"]
        is_final1 = G.nodes[n]["is_final1"]
        is_final2 = G.nodes[n]["is_final2"]
        is_joint_final = G.nodes[n]["is_joint_final"]
        in_game = G.nodes[n]["in_game"]
        if is_initial:
            return "red"
        if is_joint_final:
            return "magenta"
        if in_game == "AB":
            if is_final1 and is_final2:
                return "black"
            return "green"
        elif in_game == "A":
            if is_final1:
                return "teal"
            else:
                return "blue"
        elif in_game == "B":
            if is_final2:
                return "orange"
            else:
                return "yellow"

        return "grey"

    caption = (
        "green: both playing, blue/yellow: only one (final:teal, magenta). Initial: red. Joint final: magenta"
    )

    node_size = 3
    node_color = [color_node(_) for _ in G.nodes]
    # logger.info('layout')
    # pos = graphviz_layout(G, prog='dot')
    logger.info("drawing")

    def pos_node(n: Tuple[X, X]):
        x = G.nodes[n]["x"]
        y = G.nodes[n]["y"]
        return float(x), float(y)

    pos = {_: pos_node(_) for _ in G.nodes}

    with r.plot("s", caption=caption) as plt:
        nx.draw(
            G,
            pos=pos,
            node_color=node_color,
            cmap=plt.cm.Blues,
            arrows=False,
            edge_color=(0, 0, 0, 0.1),
            node_size=node_size,
        )
        plt.xlabel("x")
        plt.ylabel("v")
    return r
