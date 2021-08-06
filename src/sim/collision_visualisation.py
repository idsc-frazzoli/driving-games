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
        vel = vel_scale * p_report.velocity
        color = "r"
        plt.arrow(xc, yc, vel[0], vel[1], ec=color,fc=color,alpha=.8, zorder=90)
    imp_point = collision_report.impact_point.coords[0]
    plt.plot(*imp_point, "o")

    fig.set_tight_layout(True)
    plt.axis('equal')
    return
