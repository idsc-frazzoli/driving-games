import itertools
from decimal import Decimal as D
from time import perf_counter
from typing import Mapping, Dict, Set, Tuple, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from reprep import Report, MIME_GIF, MIME_PNG, RepRepDefaults, MIME_JPG, MIME_PDF
from zuper_commons.text import remove_escapes

from dg_commons import PlayerName, Color
from dg_commons.sim.simulator_animation import adjust_axes_limits
from preferences import Preference
from .game_def import Game, SolvedGameNode, GameVisualization, GamePlayer
from driving_games.metrics_structures import PlayerOutcome
from .paths import Trajectory
from .preference import PosetalPreference

from dg_commons.sim.models.vehicle import VehicleState
from .trajectory_game import (
    SolvedTrajectoryGame,
    SolvedTrajectoryGameNode,
    SolvedLeaderFollowerGame,
    SolvedRecursiveLeaderFollowerGame,
    LeaderFollowerGame,
    LeaderFollowerGameStage,
)
from .visualization import tone_down_color, TrajGameVisualization, ZOrder

EXPORT_PDF = True
STACK_JPG = False
MIME = MIME_PDF if EXPORT_PDF else MIME_JPG if STACK_JPG else MIME_PNG


def report_game_visualization(game: Game) -> Report:
    viz = game.game_vis
    r = Report("Trajectories")
    tic = perf_counter()
    with r.plot("actions") as pylab:
        ax = pylab.gca()
        with viz.plot_arena(axis=ax):
            states: List[VehicleState] = []
            for player_name, player in game.game_players.items():
                for state in player.state.support():
                    states.append(state)
                    viz.plot_player(axis=ax, player_name=player_name, state=state)
                    #todo [LEON]: are actions computed again?
                    actions = player.actions_generator.get_actions(state=state)
                    viz.plot_actions(axis=ax,
                                     actions=actions,
                                     colour=tone_down_color(player.vg.color.replace("_car", "")),
                                     width=0.5)
                    size = np.linalg.norm(ax.bbox.size) / 10000.0
                    for path in actions:
                        vals = [(x.x, x.y, x.vx) for _, x in path]
                        x, y, vel = zip(*vals)
                        ax.scatter(x, y, s=size, marker="o", c="k", alpha=0.2, zorder=ZOrder.scatter)
            ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
            pylab.axis("off")
            # uncomment if you want to plot only area around players instead of entire scene
            #adjust_axes_limits(ax=ax, plot_limits=game.game_vis.plot_limits, players_states=states)

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
                        f"\t{player}: action={action},\n" f"\t\toutcome={list(node.outcomes[player].values())}"
                    )
                texts.append("\n")
        text = "\n".join(texts)
        r_st.text(f"{k} -", remove_escapes(text))
    return r_st


def plot_outcomes_pref(
    viz: GameVisualization, axis, outcomes: PlayerOutcome, pref: Preference, pname: PlayerName, add_title: bool = True
):
    assert isinstance(pref, PosetalPreference)
    metrics: Dict[str, str] = {}
    for met in pref.graph.nodes:
        metrics[met] = str(round(float(met.evaluate(outcomes)), 2))
    viz.plot_pref(axis=axis, pref=pref, pname=pname, origin=(0.0, 0.0), labels=metrics, add_title=add_title)
    axis.set_xlim(-125.0, 125.0)
    axis.set_ylim(auto=True)


def get_stack_figure(size: Tuple[int, int]):
    rows, cols = size
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 4, rows * 4))
    if rows == 1:
        if cols == 1:
            axs = np.array([[axs]])
        else:
            axs = np.expand_dims(axs, axis=0)
    return fig, axs, set(itertools.product(range(rows), range(cols)))


def save_stack_figure(fn, fig, axs, all_idx: Set):
    for idx in all_idx:
        axs[idx].axis("off")
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    fig.savefig(fn, **RepRepDefaults.savefig_params)
    plt.close(fig=fig)


def stack_nodes(
    report: Report,
    viz: GameVisualization,
    title: str,
    players: Mapping[PlayerName, GamePlayer],
    nodes: Set[SolvedTrajectoryGameNode],
    nodes_strong: Set[SolvedTrajectoryGameNode] = None,
    plot_lead_outcomes: bool = False,
    leader: Tuple[PlayerName, Preference] = None,
):
    if plot_lead_outcomes:
        assert leader is not None

    def plot_actions(axis, sol_node: SolvedTrajectoryGameNode, width: float):
        with viz.plot_arena(axis):
            states: List[VehicleState] = []
            for pname in sol_node.actions.keys():
                for state in players[pname].state.support():
                    states.append(state)
                    viz.plot_player(axis=axis, player_name=pname, state=state)
            for pname, action in sol_node.actions.items():
                viz.plot_equilibria(
                    axis=axis,
                    actions=frozenset([action]),
                    colour=players[pname].vg.color.replace("_car",""),
                    width=width,
                    alpha=min(1.0, width),
                    ticks=False,
                    scatter=False,
                )
                axis.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
            adjust_axes_limits(ax=ax, plot_limits=viz.plot_limits, players_states=states)

    if nodes_strong is None:
        nodes_strong = set()
    with report.data_file(title, MIME) as fn:
        plot_dict = viz.get_plot_dict()
        fig, axs, all_idx = get_stack_figure((plot_dict.get_size()))
        for node in nodes:
            idx = plot_dict[node]
            ax = axs[idx]
            all_idx.remove(idx)
            if plot_lead_outcomes:
                lead, pref = leader
                plot_outcomes_pref(
                    viz=viz, axis=ax, outcomes=node.outcomes[lead], pref=pref, pname=lead, add_title=False
                )
            else:
                w: float = 2.0 if node in nodes_strong else 1.5
                plot_actions(axis=ax, sol_node=node, width=w)

        save_stack_figure(fn=fn, fig=fig, axs=axs, all_idx=all_idx)


def gif_eq(
    report: Report,
    node_eq: SolvedTrajectoryGameNode,
    game: Game,
    prefs: Mapping[PlayerName, Preference] = None,
    nash_eq: Mapping[str, SolvedTrajectoryGame] = None,
    make_gif=True,
):
    if prefs is None:
        prefs = {pname: peq.preference for pname, peq in game.game_players.items()}
    for pref in prefs.values():
        if not isinstance(pref, PosetalPreference):
            raise NotImplementedError(f"Preferences must be PosetalPreferences," f" found type {pref.get_type()}")
    eq_viz = report.figure(cols=2)
    if nash_eq is None:
        title = "Actions"
    else:
        nodes: List[str] = []
        for k, node_set_alt in nash_eq.items():
            if k == "weak":
                continue
            if node_eq in node_set_alt:
                nodes.append(k)
        title = ", ".join(nodes)

    if make_gif:
        with eq_viz.data_file(title, MIME_GIF) as fn:
            create_animation(fn=fn, game=game, node=node_eq)
        plt.close()

    with eq_viz.plot(title + "outcomes") as pylab:
        n: float = 0.0
        ax: Axes = pylab.gca()
        for pname, pref in prefs.items():
            metrics: Dict[str, str] = {}
            outcomes = node_eq.outcomes[pname]
            for met in pref.G.nodes:
                metrics[met] = str(round(float(met.evaluate(outcomes)), 2))
            game.game_vis.plot_pref(axis=ax, pref=pref, pname=pname, origin=(n, 0.0), labels=metrics)
            n = n + 200
        ax.set_xlim(-125.0, n - 125.0)
    plt.close()
    return


def report_nash_eq(game: Game, nash_eq: Mapping[str, SolvedTrajectoryGame], plot_gif: bool, max_n_gif=10) -> Report:
    tic = perf_counter()
    viz = game.game_vis
    r_all = Report("equilibria")
    req = Report("plots")

    for player in game.game_players.values():
        assert isinstance(player.preference, PosetalPreference), (
            f"Preference is of type {player.preference.get_type()} " f"and not {PosetalPreference.get_type()}"
        )

    r_all.add_child(report_states(nash_eq=nash_eq))

    node_set = nash_eq["weak"]

    def save_actions(nodes_all: Set[SolvedTrajectoryGameNode]) -> Mapping[PlayerName, Set[Trajectory]]:
        actions: Dict[PlayerName, Set[Trajectory]] = {p: set() for p in game.game_players.keys()}
        for node_eq in nodes_all:
            for player_eq, action in node_eq.actions.items():
                actions[player_eq].add(action)
        return actions

    def plot_eq_all(axis, actions_all: Mapping[PlayerName, Set[Trajectory]], w: float, color: Optional[Color] = None):
        for pname, actions in actions_all.items():
            if len(actions) == 0:
                continue
            color = game.game_players[pname].vg.colour if color is None else color
            viz.plot_equilibria(
                axis=axis,
                actions=frozenset(actions),
                colour=color,
                width=w,
                alpha=min(w, 1.0),
                scatter=True,
                plot_lanes=False,
            )

    def plot_pref(rep: Report):
        with rep.data_file("Pref", MIME) as fn:
            player = list(game.game_players.values())[0]
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
            viz.plot_pref(axis=ax, pref=player.preference, pname=player.name, origin=(0.0, 0.0), add_title=False)
            ax.set_xlim(-125.0, 125.0)
            fig.savefig(fn, **RepRepDefaults.savefig_params)
            plt.close(fig=fig)

    def image_eq(
        report: Report,
        nodes_light: SolvedTrajectoryGame,
        nodes_dark: SolvedTrajectoryGame,
        plot_actions: bool,
        plot_lanes: bool,
    ):
        eq_viz = report.figure("Overlay", cols=2)
        actions_dark = save_actions(nodes_dark)
        actions_light = save_actions(nodes_light)
        with eq_viz.plot("all_equilibria") as pylab:
            ax = pylab.gca()
            with viz.plot_arena(axis=ax):
                states: List[VehicleState] = []
                for player_name, player in game.game_players.items():
                    for state in player.state.support():
                        states.append(state)
                        viz.plot_player(axis=ax, player_name=player_name, state=state)
                        actions_all = player.actions_generator.get_actions(state=state)
                        if plot_actions:
                            viz.plot_actions(
                                axis=ax,
                                actions=actions_all,
                                colour="grey",
                                width=1.0,
                                alpha=0.7,
                                ticks=False,
                                plot_lanes=False,
                            )
                        # if plot_lanes:
                        #     lanes: Dict[DgLanelet, Optional[Polygon]] = {}
                        #     for traj in actions_all:
                        #         lane, goal = traj.get_lane()
                        #         lanes[lane] = goal
                        #     viz.plot_actions(
                        #         axis=ax, actions=frozenset(), colour=player.vg.colour, plot_lanes=True, lanes=lanes
                        #     )

                plot_eq_all(axis=ax, actions_all=actions_light, w=1.0, color="gold")
                plot_eq_all(axis=ax, actions_all=actions_dark, w=1.0, color="red")
                ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
                adjust_axes_limits(ax=ax, plot_limits=game.game_vis.plot_limits, players_states=states)
        plot_pref(rep=eq_viz)

    if plot_gif:
        i = 1
        for node in node_set:
            make_gif = i < max_n_gif
            rplot = Report(f"Eq_{i}")
            gif_eq(report=rplot, node_eq=node, game=game, nash_eq=nash_eq, make_gif=make_gif)
            req.add_child(rplot)
            i += 1
    rplot = Report(f"Equilibria:Strong-Weak")
    image_eq(report=rplot, nodes_dark=nash_eq["strong"], nodes_light=node_set, plot_actions=True, plot_lanes=False)

    if len(node_set) < 200:
        eq_viz = rplot.figure("Stacked", cols=2)
        stack_nodes(
            report=eq_viz,
            viz=viz,
            title="all_equilibria",
            players=game.game_players,
            nodes=node_set,
            nodes_strong=nash_eq["strong"],
        )
        plot_pref(rep=eq_viz)
    req.add_child(rplot)

    rplot = Report(f"Equilibria:Admissible-Weak")
    image_eq(report=rplot, nodes_dark=nash_eq["admissible"], nodes_light=node_set, plot_actions=False, plot_lanes=True)
    req.add_child(rplot)

    r_all.add_child(req)
    toc = perf_counter() - tic
    print(f"Nash eq viz time = {toc:.2f} s")
    return r_all


def report_preferences(viz: GameVisualization, players: Mapping[PlayerName, Preference]) -> Report:
    tic = perf_counter()
    r = Report("Preference_structures")

    for player, pref in players.items():
        with r.plot(player) as pylab:
            ax: Axes = pylab.gca()
            viz.plot_pref(axis=ax, pref=pref, pname=player, origin=(0.0, 0.0))
            ax.set_xlim(-125.0, 125.0)
    toc = perf_counter() - tic
    print(f"Preference viz time = {toc:.2f} s")
    return r


def create_animation(fn: str, game: Game, node: SolvedGameNode):
    viz = game.game_vis
    assert isinstance(node, SolvedTrajectoryGameNode)
    assert isinstance(viz, TrajGameVisualization)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.set_tight_layout(True)
    ax.set_aspect(1)
    box = {}

    def init_plot():
        ax.clear()
        ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        with viz.plot_arena(axis=ax):
            states: List[VehicleState] = []
            for player_name, player in game.game_players.items():
                for state in player.state.support():
                    states.append(state)
                    box[player_name] = viz.plot_player(axis=ax, player_name=player_name, state=state)
            for player, action in node.actions.items():
                viz.plot_equilibria(
                    axis=ax,
                    actions=frozenset([action]),
                    colour=game.game_players[player].vg.colour,
                    width=1.0,
                    scatter=False,
                )
            adjust_axes_limits(ax=ax, plot_limits=game.game_vis.plot_limits, players_states=[])

        return list(itertools.chain.from_iterable(box.values()))

    def update_plot(t: D):
        states: List[VehicleState] = []
        for player, box_handle in box.items():
            action: Trajectory = node.actions[player]
            state = action.at(t=t)
            states.append(state)
            box[player] = viz.plot_player(axis=ax, player_name=player, state=state, box=box_handle)
        # adjust_axes_limits(ax=ax, plot_limits=game.game_vis.plot_limits, players_states=states)
        return list(itertools.chain.from_iterable(box.values()))

    actions = list(node.actions.values())
    lens = [_.get_end() for _ in actions]
    longest = lens.index(max(lens))
    times = actions[longest].get_sampling_points()
    dt_ms = int((times[1] - times[0]) * 1000)
    anim = FuncAnimation(
        fig=fig, func=update_plot, init_func=init_plot, frames=times, interval=dt_ms, blit=True, repeat_delay=0
    )
    anim.save(fn, dpi=200, writer="imagemagick")


def report_leader_follower_solution(
    game: Game, solution: SolvedLeaderFollowerGame, plot_gif: bool, stage: int = 0
) -> Report:
    PLOT_ALL_OUT = False

    tic = perf_counter()
    title = f"Leader - follower game solutions (Stage = {stage})"
    report_all = Report(title)
    lf = solution.lf
    if stage == 0:
        report_all.text("Players:", f"Leader = {lf.leader}, Follower = {lf.follower}")
        report_all.text("Antichain_Comparison:", lf.antichain_comparison)

    # Create dictionary for leader actions and follower prefs
    i_act = 1
    act_dict: Dict[Trajectory, int] = {}
    for act in solution.games.keys():
        act_dict[act] = i_act
        i_act += 1
    pref_dict_f = game.game_vis.get_pref_dict(player=lf.follower)
    p_f_dict: Dict[Preference, int] = {}
    p_f_list: List[Optional[Preference]] = [None for _ in range(len(pref_dict_f))]
    rows, cols = pref_dict_f.get_size()
    for p_f in lf.prefs_follower_est.support():
        r, c = pref_dict_f[p_f]
        index = r * cols + c
        p_f_dict[p_f] = index + 1
        p_f_list[index] = p_f
    no_pref = PosetalPreference(pref_str="NoPreference")

    actions_text = "All leader best actions:\n" + "\n".join([f"A_{idx}: {str(act)}" for act, idx in act_dict.items()])
    report_all.text("Leader_actions:", actions_text)
    r_pref = Report("Player_Preferences")

    def stack_prefs(pname: PlayerName, pprefs: List[Preference]):
        # Plot all prefs for all players as a grid
        pviz = r_pref.figure(f"Preferences_{pname}", cols=1)
        with pviz.data_file(f"Preferences_{pname}", MIME) as fn:
            pref_dict = game.game_vis.get_pref_dict(player=pname)
            fig, axs, all_idx = get_stack_figure(size=pref_dict.get_size())
            for pref in pprefs:
                if pref is None:
                    continue
                idx = pref_dict[pref]
                ax = axs[idx]
                all_idx.remove(idx)
                game.game_vis.plot_pref(axis=ax, pref=pref, pname=pname, origin=(0.0, 0.0), add_title=False)
                ax.set_xlim(-125.0, 125.0)
                ax.set_ylim(auto=True)
            save_stack_figure(fn=fn, fig=fig, axs=axs, all_idx=all_idx)

    tic1 = perf_counter()
    stack_prefs(pname=lf.leader, pprefs=[lf.pref_leader])
    stack_prefs(pname=lf.follower, pprefs=p_f_list)
    report_all.add_child(r_pref)
    toc1 = perf_counter() - tic1

    # Print best leader actions for each comb of prefs
    rep_act = Report("Best_Leader_Actions")
    for p_f in p_f_list:
        if p_f is None:
            continue
        i_f = p_f_dict[p_f]
        actions = solution.best_leader_actions[p_f]
        text = ", ".join([f"Action_{act_dict[act]}" for act in actions])
        rep_act.text(f"{lf.follower}:Pref_{i_f}", text)
    report_all.add_child(rep_act)

    toc_br, toc_out, toc_gif = 0.0, 0.0, 0.0
    # Group plots based on leader action
    if stage == 0:
        print(
            f"Report Params:\n\tTotal leader actions = {len(solution.games)},"
            f"\n\tTotal follower preferences = {len(p_f_dict)},"
            f"\n\tPlotting gifs = {plot_gif}"
        )
    for act, sol in solution.games.items():

        # Aggregate all nodes for all prefs of follower to create grid
        # BR is not a function of p_l, so we can use any one p_l
        all_nodes: SolvedTrajectoryGame = set()
        pf_nodes: Dict[str, SolvedTrajectoryGame] = {}
        for p_f in p_f_list:
            if p_f is None:
                continue
            i_f = p_f_dict[p_f]
            nodes = sol[p_f].nodes
            all_nodes |= nodes
            pf_nodes[f"Pref_{i_f}"] = nodes
        game.game_vis.init_plot_dict(values=all_nodes)

        rep_act = Report(f"Action_{act_dict[act]}")
        rep_act.text("Action", f"Leader Trajectory = {act}")

        tic_gif = perf_counter()
        if plot_gif:
            pref_f = no_pref if lf.pref_follower_real is None else lf.pref_follower_real
            prefs = {lf.leader: lf.pref_leader, lf.follower: pref_f}
            i_gif = 1
            for node in all_nodes:
                rplot = Report(f"BR_{i_gif}")
                gif_eq(report=rplot, node_eq=node, game=game, prefs=prefs, nash_eq=pf_nodes)
                rep_act.add_child(rplot)
                i_gif += 1
        toc_gif += perf_counter() - tic_gif

        for p_f in p_f_list:
            if p_f is None:
                continue
            i_f = p_f_dict[p_f]

            tic_br = perf_counter()
            # For each pref of follower, plot grid of best responses
            rep = Report(f"{lf.follower}:Pref_{i_f}")
            rep_nodes = sol[p_f].nodes
            if not plot_gif:
                stack_viz = rep.figure("Best_Responses", cols=1)
                stack_nodes(
                    report=stack_viz, viz=game.game_vis, title=f"Solutions", players=game.game_players, nodes=rep_nodes
                )
            toc_br += perf_counter() - tic_br

            tic_out = perf_counter()
            # Plot grid of leader outcomes and aggregated outcome
            node_sols = sol[p_f]
            lead_viz = rep.figure(f"Leader_Outcomes", cols=1 + int(PLOT_ALL_OUT))
            if PLOT_ALL_OUT:
                stack_nodes(
                    report=lead_viz,
                    viz=game.game_vis,
                    title=f"{lf.leader}_outcomes",
                    players=game.game_players,
                    nodes=rep_nodes,
                    plot_lead_outcomes=True,
                    leader=(lf.leader, lf.pref_leader),
                )
            with lead_viz.plot(f"{lf.leader}_agg_outcomes") as pylab:
                plot_outcomes_pref(
                    viz=game.game_vis,
                    axis=pylab.gca(),
                    outcomes=node_sols.agg_lead_outcome,
                    pref=lf.pref_leader,
                    pname=lf.leader,
                    add_title=False,
                )
            toc_out += perf_counter() - tic_out
            rep_act.add_child(rep)
        report_all.add_child(rep_act)

    toc = perf_counter() - tic
    if stage == 0:
        print(
            f"Times:\n\tPlayer prefs viz time = {toc1:.2f} s"
            f"\n\tBest response viz time = {toc_br:.2f} s"
            f"\n\tGif viz time = {toc_gif:.2f} s"
            f"\n\tOutcomes viz time = {toc_out:.2f} s"
            f"\n\tSolutions viz time = {toc:.2f} s"
        )
    return report_all


def report_leader_follower_recursive(
    game: LeaderFollowerGame, result: SolvedRecursiveLeaderFollowerGame, plot_gif: bool
) -> Report:
    rep = Report("Leader-Follower")
    gif_viz = rep.figure(cols=1)
    with gif_viz.data_file("Solution", MIME_GIF) as fn:
        create_animation_recursive(fn=fn, game=game, result=result)

    def plot_out_player(pname: PlayerName, pref: Preference):
        with outcome_viz.plot(f"{pname}_agg_outcomes") as pylab:
            plot_outcomes_pref(
                viz=game.game_vis,
                axis=pylab.gca(),
                outcomes=result.aggregated_node.outcomes[pname],
                pref=pref,
                pname=pname,
                add_title=False,
            )

    def plot_traj_players():
        with act_viz.plot("Driven trajectories") as pylab:
            ax = pylab.gca()
            with game.game_vis.plot_arena(axis=ax):
                for pname, player in game.game_players.items():
                    for state in player.state.support():
                        game.game_vis.plot_player(axis=ax, player_name=pname, state=state)
                    p_act = frozenset([result.aggregated_node.actions[pname]])
                    game.game_vis.plot_equilibria(
                        axis=ax,
                        actions=p_act,
                        colour=game.game_players[pname].vg.colour,
                        width=1.0,
                        alpha=1.0,
                        scatter=False,
                    )

    prefs = {game.lf.leader: game.lf.pref_leader, game.lf.follower: game.lf.pref_follower_real}
    rep.add_child(report_preferences(viz=game.game_vis, players=prefs))

    outcome_viz = rep.figure(f"Overall Player Outcomes", cols=2)
    plot_out_player(pname=game.lf.leader, pref=game.lf.pref_leader)
    plot_out_player(pname=game.lf.follower, pref=game.lf.pref_follower_real)

    act_viz = rep.figure(cols=1)
    plot_traj_players()

    def update_states(sol: LeaderFollowerGameStage):
        for pname, player in game.game_players.items():
            player.state = sol.states[pname]

    times = result.stages.get_sampling_points()
    for stage in range(len(times)):
        stage_sol = result.stages.at(times[stage])
        update_states(sol=stage_sol)
        rep.add_child(
            report_leader_follower_solution(game=game, solution=stage_sol.lf_game, plot_gif=False, stage=stage)
        )
    return rep


def create_animation_recursive(fn: str, game: Game, result: SolvedRecursiveLeaderFollowerGame):
    class Counter:
        value: int

        def __init__(self):
            self.value = -1

        def inc(self):
            self.value += 1

        def get(self):
            return self.value

    viz = game.game_vis
    assert isinstance(viz, TrajGameVisualization)

    solve_times = result.stages.get_sampling_points()
    agg_actions = result.aggregated_node.actions
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.set_tight_layout(True)
    ax.set_aspect(1)
    states, actions, opt_actions = {}, {}, {}
    i_plot = Counter()

    def get_list() -> List:
        return (
            list(itertools.chain.from_iterable(states.values())) + list(actions.values()) + list(opt_actions.values())
        )

    def init_plot():
        ax.clear()
        with viz.plot_arena(axis=ax):
            sol_0 = result.stages.at(solve_times[0])
            for pname, player in game.game_players.items():
                action: Trajectory = sol_0.game_node.actions[pname]
                state = action.at(t=solve_times[0])
                states[pname] = viz.plot_player(axis=ax, state=state, player_name=pname, alpha=0.7)
                actions[pname] = viz.plot_actions(axis=ax, actions=frozenset([]), colour=player.vg.colour, width=0.3)
                opt_actions[pname] = viz.plot_actions(axis=ax, actions=frozenset([]), width=0.75)
        return get_list()

    def update_actions():
        i_plot.inc()
        i = i_plot.get()
        sol_i = result.stages.at(solve_times[i])
        for pname, player in game.game_players.items():
            colour = player.vg.colour if i == 0 else None
            p_act = sol_i.context.player_actions[pname]
            actions[pname] = viz.plot_actions(axis=ax, actions=p_act, colour=colour, lines=actions[pname])
        lead, foll = sol_i.lf.leader, sol_i.lf.follower
        l_act = frozenset([sol_i.game_node.actions[lead]])
        f_act = frozenset(sol_i.best_responses_pred)
        opt_actions[lead] = viz.plot_actions(axis=ax, actions=l_act, lines=opt_actions[lead])
        opt_actions[foll] = viz.plot_actions(axis=ax, actions=f_act, lines=opt_actions[foll])

    def update_plot(t: D):
        i = i_plot.get()
        if i < 0 or (i + 1 < len(solve_times) and t >= solve_times[i + 1]):
            update_actions()
        for pname, box_handle in states.items():
            state = agg_actions[pname].at(t=t)
            states[pname] = viz.plot_player(axis=ax, player_name=pname, state=state, box=box_handle)
        return get_list()

    times = agg_actions[result.lf.leader].get_sampling_points()
    dt_ms = int((times[1] - times[0]) * 1000)
    anim = FuncAnimation(
        fig=fig, func=update_plot, init_func=init_plot, frames=times, interval=dt_ms, blit=True, repeat_delay=0
    )
    anim.save(fn, dpi=200, writer="imagemagick")
