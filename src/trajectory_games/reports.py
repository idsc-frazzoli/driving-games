import itertools
from time import perf_counter
from typing import Mapping, Dict, Set, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from reprep import Report, MIME_GIF, MIME_PNG, RepRepDefaults
from zuper_commons.text import remove_escapes
from decimal import Decimal as D

from games import PlayerName
from preferences import Preference
from .game_def import Game, SolvedGameNode, GameVisualization, GamePlayer
from .trajectory_game import SolvedTrajectoryGame, SolvedTrajectoryGameNode, SolvedLeaderFollowerGame, \
    SolvedLeaderFollowerGameNode
from .preference import PosetalPreference
from .paths import Trajectory
from .visualization import TrajGameVisualization
from .metrics_def import PlayerOutcome


def report_game_visualization(game: Game) -> Report:
    viz = game.game_vis
    r = Report("Trajectories")
    tic = perf_counter()
    with r.plot("actions") as pylab:
        ax = pylab.gca()
        with viz.plot_arena(axis=ax):
            for player_name, player in game.game_players.items():
                for state in player.state.support():
                    viz.plot_player(axis=ax, player_name=player_name, state=state)
                    actions = player.actions_generator.get_actions_static(state=state, world=game.world,
                                                                          player=player.name)
                    viz.plot_actions(axis=ax, actions=actions, colour=player.vg.colour, width=0.5)

    toc = perf_counter() - tic
    print(f"Report game viz time = {toc:.2f} s")
    return r


def report_states(nash_eq: Mapping[str, SolvedTrajectoryGame]) -> Report:
    r_st = Report("states")
    print_all = len(nash_eq["weak"]) <= 10
    for k, node_set in nash_eq.items():
        texts = []
        if not bool(node_set):
            texts.append("\t No equilibria")
        elif not print_all or k.startswith("weak"):
            texts.append(f"\t {len(node_set)} equilibria")
        else:
            for node in node_set:
                for player, action in node.actions.items():
                    texts.append(
                        f"\t{player}: action={action},\n"
                        f"\t\toutcome={list(node.outcomes[player].values())}"
                    )
                texts.append("\n")
        text = "\n".join(texts)
        r_st.text(f"{k} -", remove_escapes(text))
    return r_st


def plot_outcomes_pref(viz: GameVisualization, axis, outcomes: PlayerOutcome,
                       pref: Preference, pname: PlayerName):
    assert isinstance(pref, PosetalPreference)
    metrics: Dict[str, str] = {}
    for met in pref.graph.nodes:
        metrics[met] = str(round(float(met.evaluate(outcomes)), 2))
    viz.plot_pref(axis=axis, pref=pref, pname=pname, origin=(0.0, 0.0), labels=metrics)
    axis.set_xlim(-150.0, 100.0)
    axis.set_ylim(auto=True)


def stack_nodes(report: Report, viz: GameVisualization, title: str,
                players: Mapping[PlayerName, GamePlayer],
                nodes: Set[SolvedTrajectoryGameNode],
                nodes_strong: Set[SolvedTrajectoryGameNode] = None,
                plot_lead_outcomes: bool = False,
                leader: Tuple[PlayerName, Preference] = None):

    if plot_lead_outcomes:
        assert leader is not None

    def plot_actions(axis, sol_node: SolvedTrajectoryGameNode, width: float):
        with viz.plot_arena(axis):
            for pname in sol_node.actions.keys():
                for state in players[pname].state.support():
                    viz.plot_player(axis=axis, player_name=pname,
                                    state=state)
            for pname, action in sol_node.actions.items():
                viz.plot_equilibria(axis=axis, actions=frozenset([action]),
                                    colour=players[pname].vg.colour,
                                    width=width, alpha=width, ticks=False)

    if nodes_strong is None:
        nodes_strong = set()
    with report.data_file(title, MIME_PNG) as fn:
        plot_dict = viz.get_plot_dict()
        rows, cols = plot_dict.get_size()
        all_idx = set(itertools.product(range(rows), range(cols)))
        fig, axs = plt.subplots(rows, cols)
        if rows == 1:
            if cols == 1:
                axs = np.array([[axs]])
            else:
                axs = np.expand_dims(axs, axis=0)
        for node in nodes:
            idx = plot_dict[node]
            ax = axs[idx]
            all_idx.remove(idx)
            if plot_lead_outcomes:
                lead, pref = leader
                plot_outcomes_pref(viz=viz, axis=ax, outcomes=node.outcomes[lead],
                                   pref=pref, pname=lead)
            else:
                w: float = 1.0 if node in nodes_strong else 0.5
                plot_actions(axis=ax, sol_node=node, width=w)

        for idx in all_idx:
            axs[idx].axis('off')
        fig.subplots_adjust(hspace=0.0, wspace=0.0)
        fig.savefig(fn, **RepRepDefaults.savefig_params)
        plt.close(fig=fig)


def report_nash_eq(game: Game, nash_eq: Mapping[str, SolvedTrajectoryGame],
                   plot_gif: bool) -> Report:
    tic = perf_counter()
    viz = game.game_vis
    r_all = Report("equilibria")
    req = Report("plots")

    for player in game.game_players.values():
        assert isinstance(player.preference, PosetalPreference), \
            f"Preference is of type {player.preference.get_type()} " \
            f"and not {PosetalPreference.get_type()}"

    r_all.add_child(report_states(nash_eq=nash_eq))

    node_set = nash_eq["weak"]

    def gif_eq(report: Report, node_eq: SolvedTrajectoryGameNode):
        eq_viz = report.figure(cols=2)
        nodes: str = ""
        for k, node_set_alt in nash_eq.items():
            if k == "weak":
                continue
            if node_eq in node_set_alt:
                nodes += f"{k}, "
        title = f"Equilibrium: ({nodes[:-2]})"

        with eq_viz.data_file(title, MIME_GIF) as fn:
            create_animation(fn=fn, game=game, node=node_eq)

        with eq_viz.plot("outcomes") as pylab:
            n: float = 0.0
            ax: Axes = pylab.gca()
            for player_name, player_eq in game.game_players.items():
                metrics: Dict[str, str] = {}
                outcomes = node_eq.outcomes[player_name]
                for pref in player_eq.preference.graph.nodes:
                    metrics[pref] = str(round(float(pref.evaluate(outcomes)), 2))
                viz.plot_pref(axis=ax, pref=player_eq.preference,
                              pname=player_eq.name, origin=(n, 0.0), labels=metrics)
                n = n + 200
            ax.set_xlim(-150.0, n - 100.0)

    def save_actions(nodes_all: Set[SolvedTrajectoryGameNode]) -> Mapping[PlayerName, Set[Trajectory]]:
        actions: Dict[PlayerName, Set[Trajectory]] = {p: set() for p in game.game_players.keys()}
        for node_eq in nodes_all:
            for player_eq, action in node_eq.actions.items():
                actions[player_eq].add(action)
        return actions

    def plot_eq_all(axis, actions_all: Mapping[PlayerName, Set[Trajectory]], w: float):
        for pname, actions in actions_all.items():
            if len(actions) == 0:
                continue
            viz.plot_equilibria(axis=axis, actions=frozenset(actions),
                                colour=game.game_players[pname].vg.colour,
                                width=w, alpha=w)

    def plot_pref(rep: Report):
        with rep.plot("pref") as pylab:
            player = list(game.game_players.values())[0]
            ax: Axes = pylab.gca()
            viz.plot_pref(axis=ax, pref=player.preference, pname=player.name, origin=(0.0, 0.0))
            ax.set_xlim(-150.0, 125.0)

    def image_eq(report: Report):
        eq_viz = report.figure(cols=2)
        nodes_strong = nash_eq["strong"]
        nodes_weak = node_set.difference(nodes_strong)
        actions_strong = save_actions(nodes_strong)
        actions_weak = save_actions(nodes_weak)
        with eq_viz.plot("all_equilibria") as pylab:
            ax = pylab.gca()
            with viz.plot_arena(axis=ax):
                for player_name, player in game.game_players.items():
                    for state in player.state.support():
                        viz.plot_player(axis=ax, player_name=player_name,
                                        state=state)
                plot_eq_all(axis=ax, actions_all=actions_weak, w=0.5)
                plot_eq_all(axis=ax, actions_all=actions_strong, w=1.0)

            plot_pref(rep=eq_viz)

    if plot_gif:
        i = 1
        for node in node_set:
            rplot = Report(f"Eq_{i}")
            gif_eq(report=rplot, node_eq=node)
            req.add_child(rplot)
            i += 1
    else:
        rplot = Report(f"Equilibria")
        if len(node_set) > 100:
            image_eq(report=rplot)
        else:
            eq_viz = rplot.figure(cols=2)
            stack_nodes(report=eq_viz, viz=viz, title="all_equilibria", players=game.game_players,
                        nodes=node_set, nodes_strong=nash_eq["strong"])
            plot_pref(rep=eq_viz)
        req.add_child(rplot)

    r_all.add_child(req)
    toc = perf_counter() - tic
    print(f"Nash eq viz time = {toc:.2f} s")
    return r_all


def report_leader_follower_solution(game: Game, solution: SolvedLeaderFollowerGame) -> Report:

    tic = perf_counter()
    report_all = Report("Leader - follower game solutions")
    lead, foll = next(iter(solution)).players
    report_all.text("Players:", f"Leader = {lead}, Follower = {foll}")
    strings = {False: "Simulated", True: "Predicted"}

    def plot_common(report: Report, sol_node: SolvedLeaderFollowerGameNode, pred: bool):
        pref_all = sol_node.player_pref.predicted if pred else sol_node.player_pref.simulated
        with report.plot(f"Leader_agg_outcomes") as pylab:
            ax: Axes = pylab.gca()
            lead_game = sol_node.leader_game
            outcomes = (lead_game.predicted if pred else lead_game.simulated).outcomes[lead]
            plot_outcomes_pref(viz=game.game_vis, axis=ax, outcomes=outcomes,
                               pref=pref_all[lead], pname=lead)

        with report.plot(f"Follower_preference") as pylab:
            ax: Axes = pylab.gca()
            game.game_vis.plot_pref(axis=ax, pref=pref_all[foll], pname=foll, origin=(0.0, 0.0))
            ax.set_xlim(-150.0, 125.0)

    print(f"Total leader actions = {len(solution)}")
    i = 1
    for sol in solution:
        rep_sol = Report(f"Action_{i}")
        rep_sol.text("Action", f"Leader Trajectory = {sol.leader_game.predicted.actions[lead]}")
        nodes: Set[SolvedTrajectoryGameNode] = (sol.games.predicted | sol.games.simulated)
        game.game_vis.init_plot_dict(values=nodes)
        for pred in [True, False]:
            rep = Report(strings[pred])
            rep_nodes = sol.games.predicted if pred else sol.games.simulated
            plot_common(report=rep, sol_node=sol, pred=pred)
            stack_viz = rep.figure("Solutions", cols=1)
            stack_nodes(report=stack_viz, viz=game.game_vis, title=f"Solutions",
                        players=game.game_players, nodes=rep_nodes)
            lead_pref = (sol.player_pref.predicted if pred else sol.player_pref.simulated)[lead]
            stack_viz = rep.figure("Outcomes", cols=1)
            stack_nodes(report=stack_viz, viz=game.game_vis, title=f"Lead_outcomes",
                        players=game.game_players, nodes=rep_nodes,
                        plot_lead_outcomes=True, leader=(lead, lead_pref))
            rep_sol.add_child(rep)
        report_all.add_child(rep_sol)
        i += 1

    toc = perf_counter() - tic
    print(f"Solutions viz time = {toc:.2f} s")
    return report_all


def report_preferences(game: Game) -> Report:
    tic = perf_counter()
    r = Report("Preference_structures")
    viz = game.game_vis

    for player in game.game_players.values():
        with r.plot(player.name) as pylab:
            ax: Axes = pylab.gca()
            viz.plot_pref(axis=ax, pref=player.preference, pname=player.name, origin=(0.0, 0.0))
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
        with viz.plot_arena(axis=ax):
            for player_name, player in game.game_players.items():
                for state in player.state.support():
                    box[player_name] = \
                        viz.plot_player(axis=ax, player_name=player_name,
                                        state=state)
            for player, action in node.actions.items():
                viz.plot_equilibria(axis=ax, actions=frozenset([action]),
                                    colour=game.game_players[player].vg.colour,
                                    width=1.0)
        return list(box.values())

    def update_plot(t: D):
        for player, box_handle in box.items():
            action: Trajectory = node.actions[player]
            state = action.at(t=t)
            box[player] = viz.plot_player(axis=ax, player_name=player,
                                          state=state, box=box_handle)
        return list(box.values())

    actions = list(node.actions.values())
    lens = [_.get_end() for _ in actions]
    longest = lens.index(max(lens))
    times = actions[longest].get_sampling_points()
    dt_ms = 2*int((times[1]-times[0])*1000)
    anim = FuncAnimation(fig=fig, func=update_plot, init_func=init_plot,
                         frames=times, interval=dt_ms, blit=True)
    anim.save(fn, dpi=80, writer="imagemagick")
