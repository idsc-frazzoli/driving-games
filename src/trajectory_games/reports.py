from time import perf_counter
from typing import Mapping, Dict

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from reprep import Report, MIME_GIF
from zuper_commons.text import remove_escapes
from decimal import Decimal as D

from .static_game import StaticGame, StaticSolvedGameNode
from .trajectory_game import SolvedTrajectoryGame, SolvedTrajectoryGameNode
from .preference import PosetalPreference
from .paths import Trajectory
from .visualization import TrajGameVisualization


def report_game_visualization(game: StaticGame) -> Report:
    viz = game.game_vis
    r = Report("Trajectories")
    tic = perf_counter()
    with r.plot("actions") as pylab:
        ax = pylab.gca()
        with viz.plot_arena(pylab, ax):
            for player_name, player in game.game_players.items():
                for state in player.state.support():
                    viz.plot_player(pylab=pylab, player_name=player_name, state=state)
                viz.plot_actions(pylab=pylab, player=player)

    toc = perf_counter() - tic
    print(f"Report game viz time = {toc:.2f} s")
    return r


def report_nash_eq(game: StaticGame, nash_eq: Mapping[str, SolvedTrajectoryGame],
                   plot_gif: bool) -> Report:
    tic = perf_counter()
    viz = game.game_vis
    r_all = Report("equilibria")
    r = Report("states")
    req = Report("plots")

    for player in game.game_players.values():
        assert isinstance(player.preference, PosetalPreference), \
            f"Preference is of type {player.preference.get_type()} " \
            f"and not {PosetalPreference.get_type()}"

    for k, node_set in nash_eq.items():
        if k.startswith("weak"):
            continue
        texts = []
        if not bool(node_set):
            texts.append("\t No equilibria")
            text = "\n".join(texts)
            r.text(f"{k} -", remove_escapes(text))
            continue
        i = 1
        for node in node_set:
            for player, action in node.actions.items():
                texts.append(
                    f"\t{player}: action={action},\n"
                    f"\t\toutcome={list(node.outcomes[player].values())}"
                )
            texts.append("\n")
            rplot = Report(f"{k}_{i}")
            eq_viz = rplot.figure(cols=2)
            if plot_gif:
                with eq_viz.data_file("actions", MIME_GIF) as fn:
                    create_animation(fn=fn, game=game, node=node)
            else:
                with eq_viz.plot("equilibrium") as pylab:
                    ax = pylab.gca()
                    with viz.plot_arena(pylab, ax):
                        for player_name, player in game.game_players.items():
                            for state in player.state.support():
                                viz.plot_player(pylab=pylab, player_name=player_name,
                                                state=state)
                        for player, action in node.actions.items():
                            viz.plot_equilibria(pylab, path=action,
                                                player=game.game_players[player])

            with eq_viz.plot("outcomes") as pylab:
                n: float = 0.0
                for player_name, player in game.game_players.items():
                    metrics: Dict[str, str] = {}
                    outcomes = node.outcomes[player_name]
                    for pref in player.preference.graph.nodes:
                        metrics[pref] = str(round(float(pref.evaluate(outcomes)), 2))
                    viz.plot_pref(pylab, player=player, origin=(n, 0.0), labels=metrics)
                    n = n + 200
                ax: Axes = pylab.gca()
                ax.set_xlim(-150.0, n - 100.0)

            req.add_child(rplot)
            i = i + 1

        text = "\n".join(texts)
        r.text(f"{k} -", remove_escapes(text))

    r_all.add_child(r)
    r_all.add_child(req)
    toc = perf_counter() - tic
    print(f"Nash eq viz time = {toc:.2f} s")
    return r_all


def report_preferences(game: StaticGame) -> Report:
    tic = perf_counter()
    r = Report("Preference_structures")
    viz = game.game_vis

    for player in game.game_players.values():
        with r.plot(player.name) as pylab:
            viz.plot_pref(pylab, player=player, origin=(0.0, 0.0))
            ax: Axes = pylab.gca()
            ax.set_xlim(-150.0, 125.0)
    toc = perf_counter() - tic
    print(f"Preference viz time = {toc:.2f} s")
    return r


def create_animation(fn: str, game: StaticGame, node: StaticSolvedGameNode):

    viz = game.game_vis
    assert isinstance(node, SolvedTrajectoryGameNode)
    assert isinstance(viz, TrajGameVisualization)

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.set_aspect(1)
    box = {}

    def init_plot():
        ax.clear()
        with viz.plot_arena(plt, ax):
            for player_name, player in game.game_players.items():
                for state in player.state.support():
                    box[player_name] = \
                        viz.plot_player(pylab=plt, player_name=player_name,
                                        state=state)
            for player, action in node.actions.items():
                viz.plot_equilibria(plt, path=action,
                                    player=game.game_players[player])
        return list(box.values())

    def update_plot(t: D):
        for player, box_handle in box.items():
            action: Trajectory = node.actions[player]
            state = action.at(t=t)
            box[player] = viz.plot_player(pylab=plt, player_name=player,
                                          state=state, box=box_handle)
        return list(box.values())

    times = list(node.actions.values())[0].get_sampling_points()
    dt_ms = 2*int((times[1]-times[0])*1000)
    anim = FuncAnimation(fig=fig, func=update_plot, init_func=init_plot,
                         frames=times, interval=dt_ms, blit=True)
    anim.save(fn, dpi=80, writer="imagemagick")
