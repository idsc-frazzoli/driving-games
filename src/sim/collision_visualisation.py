from matplotlib import pyplot as plt

from sim import CollisionReport


def plot_collision(collision_report: CollisionReport):
    fig = plt.figure()
    for player, p_report in collision_report.players.items():
        footprint = p_report.footprint
        plt.plot(*footprint.exterior.xy)
        xc, yc = footprint.centroid.coords[0]
        plt.text(xc, yc, f"{player}",
                 horizontalalignment="center", verticalalignment="center")
        vel_scale = 0.3
        vel = vel_scale * p_report.velocity[0]
        vel_after = vel_scale * p_report.velocity_after[0]
        color, width = "r", 0.01
        head_width = width * 5
        plt.arrow(xc, yc, vel[0], vel[1], width=width, head_width=head_width, ec=color, fc=color, alpha=.5, zorder=90)
        plt.arrow(xc, yc, vel_after[0], vel_after[1], width=width, head_width=head_width, ec="g", fc="g", alpha=.8,
                  zorder=91)
    imp_point = collision_report.impact_point.coords[0]
    plt.plot(*imp_point, "o", zorder=85)
    n_color = "b"
    n = collision_report.impact_normal
    plt.arrow(imp_point[0], imp_point[1], n[0], n[1], ec=n_color, fc=n_color, alpha=.9, zorder=95)

    fig.set_tight_layout(True)
    plt.axis('equal')
    return
