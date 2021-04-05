from time import perf_counter
from typing import Mapping, Dict, Set

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from reprep import Report, MIME_GIF
from zuper_commons.text import remove_escapes
from decimal import Decimal as D

from games import PlayerName
from .game_def import Game, SolvedGameNode
from .trajectory_game import SolvedTrajectoryGame, SolvedTrajectoryGameNode
from .preference import PosetalPreference
from .paths import Action
from .visualization import TrajGameVisualization


def report_game_visualization(game: Game) -> Report:
    viz = game.game_vis
    r = Report("Trajectories")
    tic = perf_counter()
    with r.plot("actions") as pylab:
        ax = pylab.gca()
        with viz.plot_arena(pylab, ax):
            for player_name, player in game.game_players.items():
                for state in player.state.support():
                    viz.plot_player(pylab=pylab, player_name=player_name, state=state)
                    actions = player.actions_generator.get_actions_static(state=state, world=game.world,
                                                                          player=player.name)
                    viz.plot_actions(pylab=pylab, actions=actions, colour=player.vg.colour, width=0.5)

    toc = perf_counter() - tic
    print(f"Report game viz time = {toc:.2f} s")
    return r


def report_nash_eq(game: Game, nash_eq: Mapping[str, SolvedTrajectoryGame],
                   plot_gif: bool) -> Report:
    tic = perf_counter()
    viz = game.game_vis
    r_all = Report("equilibria")
    r_st = Report("states")
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
            r_st.text(f"{k} -", remove_escapes(text))
            continue
        for node in node_set:
            for player, action in node.actions.items():
                texts.append(
                    f"\t{player}: action={action},\n"
                    f"\t\toutcome={list(node.outcomes[player].values())}"
                )
            texts.append("\n")
        text = "\n".join(texts)
        r_st.text(f"{k} -", remove_escapes(text))

    node_set = nash_eq["weak"]

    def gif_eq(report: Report):
        eq_viz = report.figure(cols=2)
        nodes: str = ""
        for k, node_set_alt in nash_eq.items():
            if k == "weak":
                continue
            if node in node_set_alt:
                nodes += f"{k}, "
        title = f"Equilibrium: ({nodes[:-2]})"

        with eq_viz.data_file(title, MIME_GIF) as fn:
            create_animation(fn=fn, game=game, node=node)

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

    def save_actions(nodes_all: Set[SolvedTrajectoryGameNode]) -> Mapping[PlayerName, Set[Action]]:
        actions: Dict[PlayerName, Set[Action]] = {p: set() for p in game.game_players.keys()}
        for node in nodes_all:
            for player, action in node.actions.items():
                actions[player].add(action)
        return actions

    def plot_eq(pylab, actions_all: Mapping[PlayerName, Set[Action]], w: float):
        for pname, actions in actions_all.items():
            if len(actions) == 0:
                continue
            viz.plot_equilibria(pylab, actions=frozenset(actions),
                                colour=game.game_players[pname].vg.colour,
                                width=w, alpha=w)

    def image_eq(report: Report):
        nodes_strong = nash_eq["strong"]
        nodes_weak = node_set.difference(nodes_strong)
        actions_strong = save_actions(nodes_strong)
        actions_weak = save_actions(nodes_weak)
        with report.plot("all_equilibria") as pylab:
            ax = pylab.gca()
            with viz.plot_arena(pylab, ax):
                for player_name, player in game.game_players.items():
                    for state in player.state.support():
                        viz.plot_player(pylab=pylab, player_name=player_name,
                                        state=state)
                plot_eq(pylab=pylab, actions_all=actions_strong, w=1.0)
                plot_eq(pylab=pylab, actions_all=actions_weak, w=0.5)

    if plot_gif:
        i = 1
        for node in node_set:
            rplot = Report(f"Eq_{i}")
            gif_eq(report=rplot)
            req.add_child(rplot)
            i += 1
    else:
        rplot = Report(f"Equilibria")
        image_eq(report=rplot)
        req.add_child(rplot)

    r_all.add_child(r_st)
    r_all.add_child(req)
    toc = perf_counter() - tic
    print(f"Nash eq viz time = {toc:.2f} s")
    return r_all


def report_preferences(game: Game) -> Report:
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


def create_animation(fn: str, game: Game, node: SolvedGameNode):

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
                viz.plot_equilibria(plt, actions=frozenset([action]),
                                    colour=game.game_players[player].vg.colour,
                                    width=1.0)
        return list(box.values())

    def update_plot(t: D):
        for player, box_handle in box.items():
            action: Action = node.actions[player]
            state = action.at(t=t)
            box[player] = viz.plot_player(pylab=plt, player_name=player,
                                          state=state, box=box_handle)
        return list(box.values())

    actions = list(node.actions.values())
    lens = [_.get_end() for _ in actions]
    longest = lens.index(max(lens))
    times, _ = actions[longest].get_sampled_trajectory()
    dt_ms = 2*int((times[1]-times[0])*1000)
    anim = FuncAnimation(fig=fig, func=update_plot, init_func=init_plot,
                         frames=times, interval=dt_ms, blit=True)
    anim.save(fn, dpi=80, writer="imagemagick")
