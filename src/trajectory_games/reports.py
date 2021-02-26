from time import perf_counter
from typing import Mapping, Dict

from matplotlib.axes import Axes
from reprep import Report
from zuper_commons.text import remove_escapes

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
                    viz.plot_player(pylab=pylab, player_name=player_name, state=state)
                viz.plot_actions(pylab=pylab, player=player)

    toc = perf_counter() - tic
    print(f"Report game viz time = {toc:.2f} s")
    return r


def report_nash_eq(game: StaticGame, nash_eq: Mapping[str, SolvedTrajectoryGame]) -> Report:

    tic = perf_counter()
    viz = game.game_vis
    r_all = Report("equilibria")
    r = Report("states")
    req = Report("plots")

    for player in game.game_players.values():
        assert isinstance(
            player.preference, PosetalPreference
        ), f"Preference is of type {player.preference.get_type()} and not {PosetalPreference.get_type()}"

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
                    f"\t{player}: action={action},\n" f"\t\toutcome={list(node.outcomes[player].values())}"
                )
            texts.append("\n")
            rplot = Report(f"{k}_{i}")
            # TODO[SIR]: Can we reuse the same plot instead of creating new?
            with rplot.plot("equilibrium") as pylab:
                ax = pylab.gca()
                with viz.plot_arena(pylab, ax):
                    for player_name, player in game.game_players.items():
                        for state in player.state.support():
                            viz.plot_player(pylab=pylab, player_name=player_name, state=state)
                    for player, action in node.actions.items():
                        viz.plot_equilibria(pylab, path=action, player=game.game_players[player])
            with rplot.plot("preferences") as pylab:
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
    r = Report("pref")
    viz = game.game_vis

    with r.plot("preferences") as pylab:
        i: float = 0.0
        for player in game.game_players.values():
            viz.plot_pref(pylab, player=player, origin=(i, 0.0))
            i = i + 250
        ax: Axes = pylab.gca()
        ax.set_xlim(-150.0, i - 100.0)
    toc = perf_counter() - tic
    print(f"Preference viz time = {toc:.2f} s")
    return r
