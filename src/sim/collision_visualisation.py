from enum import IntEnum

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap

from sim import CollisionReport


class _Zorders(IntEnum):
    PLAYER_NAME = 90
    VEL_BEFORE = 80
    VEL_AFTER = 81
    IMPACT_LOCATION = 40
    IMPACT_LOCATION_NAME = 40
    IMPACT_POINT = 82
    IMPACT_NORMAL = 85
    DEBUG = 100


def plot_collision(collision_report: CollisionReport):
    fig = plt.figure()

    # common impact point and normals
    imp_point = collision_report.impact_point.coords[0]
    n = 0.2*collision_report.impact_normal
    plt.plot(*imp_point, "o", zorder=_Zorders.IMPACT_POINT)
    n_color = "r"
    plt.arrow(imp_point[0], imp_point[1], n[0], n[1], ec=n_color, fc=n_color, alpha=.9, zorder=_Zorders.IMPACT_NORMAL)
    # players
    name = "Dark2"
    cmap: ListedColormap = get_cmap(name)
    colors = list(cmap.colors)
    for i, (player, p_report) in enumerate(collision_report.players.items()):
        p_color = colors[i]
        # vehicle outline
        footprint = p_report.footprint
        plt.plot(*footprint.exterior.xy, color=p_color)
        xc, yc = footprint.centroid.coords[0]
        plt.text(xc, yc, f"{player}", horizontalalignment="center", verticalalignment="center",
                 zorder=_Zorders.PLAYER_NAME)
        # velocity vectors
        vel_scale = 0.3
        vel = vel_scale * p_report.velocity[0]
        vel_after = vel_scale * p_report.velocity_after[0]
        col_befor, col_after, width = "darkorange", "seagreen", 0.01
        head_width = width * 5
        plt.arrow(xc, yc, vel[0], vel[1], width=width, head_width=head_width, ec=col_befor, fc=col_befor, alpha=.8,
                  zorder=_Zorders.VEL_BEFORE)
        plt.arrow(xc, yc, vel_after[0], vel_after[1], width=width, head_width=head_width, ec=col_after, fc=col_after,
                  alpha=.8, zorder=_Zorders.VEL_AFTER)
        # todo rotational velocity
        # DEBUG velocities at collision point
        ap = np.array(imp_point) - np.array([xc, yc])
        omega = p_report.velocity[1]
        vel_atP = vel + vel_scale * (omega * ap)
        plt.arrow(imp_point[0], imp_point[1], vel_atP[0], vel_atP[1], width=width, head_width=head_width, ec=p_color,
                  fc=p_color, alpha=.8, zorder=_Zorders.DEBUG)
        # impact locations
        for loc in p_report.locations:
            loc_str, loc_shape = loc
            plt.fill(*loc_shape.exterior.xy, fc="cyan", ec="darkblue", alpha=0.4, zorder=_Zorders.IMPACT_LOCATION)
            xc, yc = loc_shape.centroid.coords[0]
            plt.text(xc, yc, f"{loc_str}", horizontalalignment="center", verticalalignment="center",
                     zorder=_Zorders.IMPACT_LOCATION_NAME)

    fig.set_tight_layout(True)
    plt.axis('equal')
    return
