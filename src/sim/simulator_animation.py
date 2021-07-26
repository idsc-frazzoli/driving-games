import math
from typing import MutableMapping, Mapping, List, Union

from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt

from games import PlayerName
from sim import logger
from sim.simulator import SimContext
from sim.simulator_structures import LogEntry
from sim.simulator_visualisation import SimVisualisation


def create_animation(file_path: str,
                     sim_context: SimContext,
                     fig_size: Union[list, None] = None,
                     dt: float = 30,
                     dpi: int = 120) -> None:
    """
    Creates an animation

    :param sim_context:
    :param file_path: filename of generated video (ends on .mp4/.gif/.avi, default mp4, when nothing is specified)
    :param fig_size: size of the video
    :param dt: time step between frames in ms
    :param dpi: resolution of the video
    :return: None
    """
    logger.info("Creating animation...")
    sim_viz: SimVisualisation = SimVisualisation(sim_context)
    time_begin = sim_context.log.get_init_time()
    time_end = sim_context.log.get_last_time()
    if not time_begin < time_end:
        raise ValueError(f"Begin time {time_begin} cannot be greater than end time {time_end}")
    if fig_size is None:
        fig_size = [15, 8]
    fig, ax = plt.subplots(figsize=fig_size)
    fig.set_tight_layout(True)
    ax.set_aspect('equal')
    states, actions, opt_actions = {}, {}, {}

    # self.f.set_size_inches(*fig_size)
    def _get_list() -> List:
        # fixme this is supposed to be an iterable of artists
        return list(states.values()) + list(actions.values()) + list(opt_actions.values())

    def init_plot():
        ax.clear()
        logger.info("Init plotting")
        with sim_viz.plot_arena(ax=ax):
            init_state: MutableMapping[PlayerName, LogEntry] = sim_context.log[time_begin]
            for pname, player in init_state.items():
                states[pname] = sim_viz.plot_player(
                    ax=ax,
                    state=player.state,
                    player_name=pname,
                    alpha=0.7)
        return _get_list()

    def update_plot(frame: int = 0):
        t: float = (frame * dt / 1000.0)
        logger.debug(f"Plotting t = {t}")
        log_at_t: Mapping[PlayerName, LogEntry] = sim_context.log.at(t)
        for pname, box_handle in states.items():
            states[pname] = sim_viz.plot_player(
                ax=ax,
                player_name=pname,
                state=log_at_t[pname].state,
                box=box_handle)

        return _get_list()

    # Min frame rate is 1 fps
    dt = min(1000.0, dt)
    frame_count: int = int(float(time_end - time_begin) // (dt / 1000.0))
    plt.ioff()
    # Interval determines the duration of each frame in ms
    anim = FuncAnimation(
        fig=fig, func=update_plot, init_func=init_plot,
        frames=frame_count, blit=False, interval=dt)

    if not any([file_path.endswith('.mp4'), file_path.endswith('.gif'), file_path.endswith('.avi')]):
        file_path += '.mp4'
    fps = int(math.ceil(1000.0 / dt))
    interval_seconds = dt / 1000.0
    anim.save(file_path, dpi=dpi, writer='ffmpeg', fps=fps,
              # extra_args=["-g", "1", "-keyint_min", str(interval_seconds)]
              )
    logger.info("Animation saved...")
    ax.clear()
