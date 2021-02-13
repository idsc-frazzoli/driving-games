from time import perf_counter
from typing import Mapping, Tuple

import networkx as nx
from reprep import Report
from zuper_commons.text import remove_escapes

from games.game_def import PlayerName
from .static_game import StaticGame
from .trajectory_game import SolvedTrajectoryGame
from .preference import PosetalPreference


def report_game_visualization(game: StaticGame) -> Report:
    viz = game.game_vis
    r = Report("vis")
    tic = perf_counter()
    with r.plot("actions") as pylab:
        ax = pylab.gca()
        with viz.plot_arena(pylab, ax):
            for player_name, player in game.game_players.items():
                for state in player.state.support():
                    viz.plot_player(player_name=player_name, state=state)
                viz.plot_actions(player=player)

    toc = perf_counter() - tic
    print(f"Report game viz time = {toc} s")
    return r


def report_nash_eq(nash_eq: Mapping[str, SolvedTrajectoryGame]) -> Report:
    # for k, node_set in nash_eq.items():
    #     print(f"\n{k} -")
    #     if not bool(node_set):
    #         print("\t No equilibria")
    #         continue
    #     for node in node_set:
    #         for player, action in node.actions.items():
    #             print(f"\t{player}: action={action},\n"
    #                   f"\t\toutcome={list(node.outcomes[player].values())}")
    #         print("\n")

    r = Report("states")
    for k, node_set in nash_eq.items():
        texts = []
        if not bool(node_set):
            texts.append("\t No equilibria")
            text = "\n".join(texts)
            r.text(f"{k} -", remove_escapes(text))
            continue
        for node in node_set:
            for player, action in node.actions.items():
                texts.append(f"\t{player}: action={action},\n"
                             f"\t\toutcome={list(node.outcomes[player].values())}")
            texts.append("\n")
        text = "\n".join(texts)
        r.text(f"{k} -", remove_escapes(text))

    # for k, node_set in nash_eq.items():
    #     # print(f"\n{k} -")
    #     if not bool(node_set):
    #         # print("\t No equilibria")
    #         continue
    #     texts = list(map(debug_print, node_set))
    #     text = "\n".join(texts)
    #     r.text(f"{k}-", remove_escapes(text))
        # for node in node_set:
        #     for player, action in node.actions.items():
        #         print(f"\t{player}: action={action},\n"
        #               f"\t\toutcome={list(node.outcomes[player].values())}")
        #     print("\n")
    return r


def report_preferences(pref_map: Mapping[PlayerName, PosetalPreference]) -> Report:
    r = Report("pref")

    with r.plot("preferences") as pylab:
        i: float = 0.0
        for player_name, pref in pref_map.items():
            plot_player(pylab, pref.graph, (i, 0.0))
            i = i + 100
    return r


def plot_player(pylab, G: nx.DiGraph, origin: Tuple[float, float]):

    X,Y = origin

    def pos_node(n: str):
        x = G.nodes[n]["x"]
        y = G.nodes[n]["y"]
        return float(x)+X, float(y)+Y

    pos = {_: pos_node(_) for _ in G.nodes}

    labels = {n:n for n in G.nodes}
    nx.draw_networkx_edges(
        G,
        pos=pos,
        edgelist=G.edges(),
        arrows=True,
    )

    ax = pylab.gca()
    nx.draw_networkx_labels(
        G,
        pos=pos,
        labels=labels,
        ax=ax,
        font_size=8,
        font_color='b'
    )
