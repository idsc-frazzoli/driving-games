import random
from typing import List, Tuple

import networkx as nx
from networkx import convert_node_labels_to_integers

from games.solve.solution_structures import GamePreprocessed
from reprep import MIME_GRAPHML, Report
from zuper_commons.text import remove_escapes
from zuper_typing import debug_print
from . import logger
from .game_def import Game, JointState, RJ, RP, SR, U, X, Y
from .reports_player import report_player

__all__ = [
    "create_report_preprocessed",
    "report_game_visualization",
    "report_game_joint_final",
]


def create_report_preprocessed(game_name: str, game_pre: GamePreprocessed) -> Report:
    r = Report(nid=game_name)
    for player_name, player in game_pre.game.players.items():
        r.add_child(report_player(game_pre, player_name, player))
        # break  # only one
    r.add_child(report_game(game_pre))
    r.add_child(report_game_joint_final(game_pre))
    return r


def report_game_visualization(game: Game) -> Report:
    """Report with the initial status of the game"""
    viz = game.game_visualization
    r = Report("vis")
    with r.plot("initial") as pylab:
        ax = pylab.gca()
        with viz.plot_arena(pylab, ax):
            for player_name, player in game.players.items():
                for x in player.initial.support():
                    viz.plot_player(player_name, state=x, commands=None, t=0)

    return r


def report_game_joint_final(game_pre: GamePreprocessed) -> Report:
    r = Report(nid="some_states", caption="Some interesting states.")
    G = game_pre.game_graph_nx

    terminal = [node for node in G if G.nodes[node]["is_terminal"]]
    terminal = random.sample(terminal, 5)
    visualize_states(game_pre, r, "terminal", terminal, "Some terminal nodes for everyone")

    pers_final = [node for node in G if G.nodes[node]["is_pers_final"]]
    pers_final = random.sample(pers_final, 5)
    visualize_states(game_pre, r, "pers_final", pers_final, "Some personal final nodes.")

    joint_final = [node for node in G if G.nodes[node]["is_joint_final_for"]]
    joint_final = random.sample(joint_final, 5)
    visualize_states(game_pre, r, "joint_final", joint_final, "Some final joint nodes.")

    return r


def visualize_states(
    game_pre: GamePreprocessed[X, U, Y, RP, RJ, SR],
    r: Report,
    name: str,
    nodes: List[JointState],
    caption: str,
):
    viz = game_pre.game.game_visualization
    f = r.figure(name, caption=caption)
    for i, node in enumerate(nodes):
        c = remove_escapes("")
        with f.plot(f"f{i}", caption=c) as pylab:
            ax = pylab.gca()
            with viz.plot_arena(pylab, ax):
                for player_name, player_state in node.items():
                    if player_state is not None:
                        viz.plot_player(player_name, state=player_state, commands=None, t=0)
    texts = list(map(debug_print, nodes))

    text = "\n".join(texts)
    r.text(f"{name}-states", remove_escapes(text))
    return f


def report_game(game_pre: GamePreprocessed) -> Report:
    G = game_pre.game_graph_nx

    r = Report(nid="game")

    with r.data_file("game", mime=MIME_GRAPHML) as fn:
        logger.info(f"done writing {fn}")
        G2 = convert_node_labels_to_integers(G)
        for (n1, n2, d) in G2.edges(data=True):
            d.clear()
        nx.write_graphml(G2, fn)

    def color_node(n):
        is_initial = G.nodes[n]["is_initial"]
        is_joint_final_for = G.nodes[n]["is_joint_final_for"]
        is_pers_final_for = G.nodes[n]["is_pers_final"]
        is_terminal = G.nodes[n]["is_terminal"]
        in_game = G.nodes[n]["in_game"]
        # todo fix terminal without collisions
        if is_initial:
            return "red"
        elif is_joint_final_for:
            return "brown"
        elif is_terminal:
            return "purple"
        elif is_pers_final_for:
            return "yellow"
        else:
            return "green"

    caption = (
        "red: initial;\n"
        "green: everyone is playing;\n"
        "brown: jointly final for someone;\n"
        "yellow: some personal one finish;\n"
        "purple: all players end."
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
