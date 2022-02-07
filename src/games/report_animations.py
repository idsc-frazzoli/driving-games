from bisect import bisect_right
from typing import Optional, Dict

from reprep import MIME_GIF, Report
from zuper_commons.text import remove_escapes
from zuper_typing import debug_print

from dg_commons import UndefinedAtTime, PlayerName
from driving_games import VehicleTrackState
from games.solve.solution_structures import GamePreprocessed, Solutions
from . import logger
from .game_def import JointState, RJ, RP, U, X, Y, SR
from .simulate import Simulation

__all__ = ["report_solutions"]


def report_solutions(gp: GamePreprocessed[X, U, Y, RP, RJ, SR], s: Solutions[X, U, Y, RP, RJ, SR]):
    r = Report()

    sims = dict(s.sims)

    f = r.figure("sequential", cols=5)
    for k, sim in list(sims.items()):
        if "follows" not in k:
            continue
        logger.info(f"drawing episode {k!r}")
        with f.data_file((k), MIME_GIF) as fn:
            create_log_animation(gp, sim, fn=fn)
        write_states(r, k, sim)
        sims.pop(k)

    f = r.figure("joint", cols=5)
    sim: Simulation
    for k, sim in list(sims.items()):
        if "joint" not in k:
            continue
        logger.info(f"drawing episode {k!r}")
        with f.data_file((k), MIME_GIF) as fn:
            create_log_animation(gp, sim, fn=fn)
        write_states(r, k, sim)
        sims.pop(k)
    js: JointState
    for i, js in enumerate(s.game_solution.initials):
        st = remove_escapes(debug_print(js))
        st += ":\n" + remove_escapes(debug_print(s.game_solution.states_to_solution[js].va.game_value))
        r.text(f"joint_st{i}", st)

    return r


def write_states(r: Report, k: str, sim: Simulation):
    texts = [f"{j}: {debug_print(v)}" for j, v in sim.states.__iter__()]
    text = "\n".join(texts)

    r.text(f"{k}-states", remove_escapes(text))

    texts = [f"{j}: {debug_print(v)}" for j, v in sim.actions.__iter__()]
    text = "\n".join(texts)
    r.text(f"{k}-actions", remove_escapes(text))

    texts = [f"{j}: {debug_print(v)}" for j, v in sim.costs.__iter__()]
    text = "\n".join(texts)
    r.text(f"{k}-costs", remove_escapes(text))

    texts = [f"{j}: {debug_print(v)}" for j, v in sim.joint_costs.__iter__()]
    text = "\n".join(texts)
    r.text(f"{k}-joint_costs", remove_escapes(text))


#
# def report_animation(
#     gp: GamePreprocessed[X, U, Y, RP, RJ, SR], sim: Simulation[X, U, Y, RP, RJ]
# ) -> Report:
#     r = Report()
#     f = r.figure()
#     with f.data_file("sim", MIME_GIF) as fn:
#         create_log_animation(gp, sim, fn=fn, upsample_log=None)
#
#     if False:
#         with f.data_file("upsampled", MIME_GIF) as fn:
#             create_log_animation(gp, sim, fn=fn, upsample_log=8)
#
#     return r


def upsample(gp, states0, actions0, n: int):
    states2 = {}
    dt = gp.solver_params.game_dt
    dt2 = dt / n
    for i, (t, s0) in enumerate(states0.items()):

        states2[t] = s0

        if i == len(states0) - 1:
            break
        actions = actions0[t]

        # logger.info('original', i=i, t=t, actions=actions)

        prev_state = s0
        for _ in range(n - 1):
            next_state = get_next_state(gp, prev_state, actions, dt2=dt2)
            this_t = t + (_ + 1) * dt2
            assert this_t not in states2
            states2[this_t] = next_state
            # logger.info('this_t', _=_, this_t=this_t)
            prev_state = next_state

    return states2


def get_next_state(gp, s0, actions, dt2):
    next_state = {}
    for player_name, action in actions.items():
        player_state = s0[player_name]
        dynamics = gp.game.malliaris[player_name].dynamics
        suc = dynamics.successors(player_state, dt2)

        if not action in suc:
            logger.info(suc=suc, action=action)
            next_state[player_name] = player_state
        else:
            successors = suc[action]

            next_state[player_name] = list(successors)[0]
    return next_state


def create_log_animation(
    gp: GamePreprocessed[X, U, Y, RP, RJ, SR],
    sim: Simulation[X, U, Y, RP, RJ],
    fn: str,
    frame_period: Optional[int] = 100,  # todo upsample to default ms between frames
):
    """
    :param gp: game preprocessed
    :param sim: simulation log
    :param fn: filename
    :param frame_period: : in ms (50 means 20Hz)
    :return:
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.set_aspect("equal")
    viz = gp.game.game_visualization

    def update(frame: int = 0):
        t: float = frame * frame_period / 1000.0
        logger.info(f"plotting t = {t}")
        ax.clear()

        # interpolating between states
        interpolated: Dict[PlayerName, VehicleTrackState]
        try:
            interpolated = sim.states.at(t)
        except UndefinedAtTime:
            ts = sim.states.timestamps
            i = bisect_right(ts, t)
            scale = (float(t) - float(ts[i - 1])) / (float(ts[i] - ts[i - 1]))
            next_js = sim.states.values[i]
            previous_js = sim.states.values[i - 1]
            interpolated = {}
            for p in previous_js:
                if p in next_js:
                    interpolated[p] = previous_js[p] * (1 - scale) + next_js[p] * scale
                else:
                    continue

        with viz.plot_arena(plt, ax):
            for player_name, player_state in interpolated.items():
                if player_state is not None:
                    viz.plot_player(player_name, state=player_state, commands=None, t=t)
        ax.set_title(f"t = {t}")
        return []

    # noinspection PyTypeChecker
    time_begin, time_end = sim.states.get_start(), sim.states.get_end()
    frame_count: int = int(float(time_end - time_begin) // (frame_period / 1000.0))
    anim = FuncAnimation(fig, func=update, frames=frame_count, blit=True, interval=frame_period)
    anim.save(fn, dpi=80, writer="imagemagick")
