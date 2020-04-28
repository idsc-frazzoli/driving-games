from decimal import Decimal as D

from reprep import MIME_GIF, Report

from games import GamePreprocessed, Optional
from games.game_def import X, U, Y, RP, RJ
from games.simulate import Simulation
from driving_games import logger
from games.solution import Solutions


def report_solutions(gp: GamePreprocessed[X, U, Y, RP, RJ], s: Solutions[X, U, Y, RP, RJ]):
    r = Report()
    for k, sim in s.sims.items():
        f = r.figure(k)
        with f.data_file("sim", MIME_GIF) as fn:
            create_log_animation(gp, sim, fn=fn, upsample_log=None)

    return r


def report_animation(
    gp: GamePreprocessed[X, U, Y, RP, RJ], sim: Simulation[X, U, Y, RP, RJ]
) -> Report:
    r = Report()
    f = r.figure()
    with f.data_file("sim", MIME_GIF) as fn:
        create_log_animation(gp, sim, fn=fn, upsample_log=None)

    if False:
        with f.data_file("upsampled", MIME_GIF) as fn:
            create_log_animation(gp, sim, fn=fn, upsample_log=8)

    return r


def upsample(gp, states0, actions0, n: int):
    states2 = {}
    dt = gp.dt
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
        dynamics = gp.game.players[player_name].dynamics
        suc = dynamics.successors(player_state, dt2)

        if not action in suc:
            logger.info(suc=suc, action=action)
            next_state[player_name] = player_state
        else:
            successors = suc[action]
            next_state[player_name] = list(successors)[0]
    return next_state


def create_log_animation(
    gp: GamePreprocessed[X, U, Y, RP, RJ],
    sim: Simulation[X, U, Y, RP, RJ],
    fn: str,
    upsample_log: Optional[int],
):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.set_aspect(1)

    interval0 = 400
    if upsample_log:

        interval = interval0 / upsample_log
        use_states = upsample(gp, sim.states, sim.actions, n=upsample_log)

    else:
        interval = interval0
        use_states = sim.states

    viz = gp.game.game_visualization

    frames = list(use_states)

    def update(t: D):
        logger.info(f"plotting t = {t}")
        ax.clear()
        states = use_states[t]
        with viz.plot_arena(plt, ax):
            for player_name, player_state in states.items():
                if player_state is not None:
                    viz.plot_player(player_name, state=player_state, commands=None)
        ax.set_title(f"t = {t}")

    # noinspection PyTypeChecker
    anim = FuncAnimation(fig, func=update, frames=frames, interval=interval)
    anim.save(fn, dpi=80, writer="imagemagick")
