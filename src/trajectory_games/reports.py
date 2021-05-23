import itertools
from time import perf_counter
from typing import Mapping, Dict, Set, Tuple, List

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
from .trajectory_game import SolvedTrajectoryGame, SolvedTrajectoryGameNode, SolvedLeaderFollowerGame
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

    PLOT_ALL_OUT = False

    tic = perf_counter()
    report_all = Report("Leader - follower game solutions")
    lf = solution.lf
    p_l_0 = lf.prefs_leader[0]
    report_all.text("Players:", f"Leader = {lf.leader}, Follower = {lf.follower}")

    # Create dictionary for leader actions
    i_act = 1
    act_dict: Dict[Trajectory, int] = {}
    for act in solution.games.keys():
        act_dict[act] = i_act
        i_act += 1

    actions_text = "All leader best actions:\n" + \
                   "\n".join([f"A_{idx}: {str(act)}" for act, idx in act_dict.items()])
    report_all.text("Leader_actions:", actions_text)
    r_pref = Report("Player_Preferences")

    def stack_prefs(pname: PlayerName, prefs: List[Preference]):
        # Plot all prefs for all players as a grid
        pviz = r_pref.figure(f"Preferences_{pname}", cols=len(prefs))
        idx = 1
        for pref in prefs:
            with pviz.plot(f"{pname}:Pref_{idx}") as pylab:
                ax: Axes = pylab.gca()
                game.game_vis.plot_pref(axis=ax, pref=pref, pname=pname, origin=(0.0, 0.0))
                ax.set_xlim(-150.0, 125.0)
            idx += 1

    tic1 = perf_counter()
    stack_prefs(pname=lf.leader, prefs=lf.prefs_leader)
    stack_prefs(pname=lf.follower, prefs=lf.prefs_follower)
    report_all.add_child(r_pref)
    toc1 = perf_counter() - tic1
    print(f"Player prefs viz time = {toc1:.2f} s")

    # Print best leader actions for each comb of prefs
    i_l = 1
    rep_act = Report("Best_Leader_Actions")
    for p_l in lf.prefs_leader:
        i_f = 1
        for p_f in lf.prefs_follower:
            actions = solution.best_leader_actions[(p_l, p_f)]
            text = f"{lf.leader}:Pref_{i_l}, {lf.follower}:Pref_{i_f}\n\t" + "{" +\
                   ", ".join([f"A_{act_dict[act]}" for act in actions]) + "}"
            rep_act.text(f"Act_{i_l}_{i_f}", text)
            i_f += 1
        i_l += 1
    report_all.add_child(rep_act)

    toc_br, toc_out = 0.0, 0.0
    # Group plots based on leader action
    print(f"Total leader actions = {len(solution.games)}")
    for act, sol in solution.games.items():

        # Aggregate all nodes for all prefs of follower to create grid
        # BR is not a function of p_l, so we can use any one p_l
        all_nodes: SolvedTrajectoryGame = set()
        for p_f in lf.prefs_follower:
            all_nodes |= sol[(p_l_0, p_f)].nodes
        game.game_vis.init_plot_dict(values=all_nodes)

        rep_act = Report(f"Action_{act_dict[act]}")
        rep_act.text("Action", f"Leader Trajectory = {act}")

        i_pf = 1
        for p_f in lf.prefs_follower:

            tic_br = perf_counter()
            # For each pref of follower, plot grid of best responses
            rep = Report(f"{lf.follower}:Pref_{i_pf}")
            rep_nodes = sol[(p_l_0, p_f)].nodes
            stack_viz = rep.figure("Best_Responses", cols=1)
            stack_nodes(report=stack_viz, viz=game.game_vis, title=f"Solutions",
                        players=game.game_players, nodes=rep_nodes)
            toc_br += perf_counter() - tic_br

            i_pl = 1
            for p_l in lf.prefs_leader:

                tic_out = perf_counter()
                # For each pref of leader, plot grid of leader outcomes and aggregated outcome
                node_sols = sol[(p_l, p_f)]
                lead_viz = rep.figure(f"{lf.leader}:Pref_{i_pl}", cols=1+int(PLOT_ALL_OUT))
                if PLOT_ALL_OUT:
                    stack_nodes(report=lead_viz, viz=game.game_vis, title=f"{lf.leader}_outcomes",
                                players=game.game_players, nodes=rep_nodes,
                                plot_lead_outcomes=True, leader=(lf.leader, p_l))
                with lead_viz.plot(f"{lf.leader}_agg_outcomes") as pylab:
                    plot_outcomes_pref(viz=game.game_vis, axis=pylab.gca(),
                                       outcomes=node_sols.agg_lead_outcome,
                                       pref=p_l, pname=lf.leader)
                toc_out += perf_counter() - tic_out
                i_pl += 1
            rep_act.add_child(rep)
            i_pf += 1
        report_all.add_child(rep_act)

    print(f"Best response viz time = {toc_br:.2f} s")
    print(f"Outcomes viz time = {toc_out:.2f} s")
    toc = perf_counter() - tic
    print(f"Solutions viz time = {toc:.2f} s")
    return report_all


def report_preferences(viz: GameVisualization, players: Mapping[PlayerName, Preference]) -> Report:
    tic = perf_counter()
    r = Report("Preference_structures")

    for player, pref in players.items():
        with r.plot(player) as pylab:
            ax: Axes = pylab.gca()
            viz.plot_pref(axis=ax, pref=pref, pname=player, origin=(0.0, 0.0))
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
