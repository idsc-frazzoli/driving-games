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
        vel = p_report.velocity
        velocity_arrow = ((xc, yc), (xc + vel[0], yc + vel[1]))
        plt.plot(velocity_arrow, "r-", zorder=90)

    fig.set_tight_layout(True)
    plt.axis('equal')
    return
