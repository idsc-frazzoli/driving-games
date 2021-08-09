from matplotlib import pyplot as plt

from sim import CollisionReport


def plot_collision(collision_report: CollisionReport):
    fig = plt.figure()
    for player, p_report in collision_report.players.items():
        # vehicle outline
        footprint = p_report.footprint
        plt.plot(*footprint.exterior.xy)
        xc, yc = footprint.centroid.coords[0]
        plt.text(xc, yc, f"{player}", horizontalalignment="center", verticalalignment="center", zorder=80)
        # velocity vectors
        vel_scale = 0.3
        vel = vel_scale * p_report.velocity[0]
        vel_after = vel_scale * p_report.velocity_after[0]
        color, width = "r", 0.01
        head_width = width * 5
        plt.arrow(xc, yc, vel[0], vel[1], width=width, head_width=head_width, ec=color, fc=color, alpha=.5, zorder=80)
        plt.arrow(xc, yc, vel_after[0], vel_after[1], width=width, head_width=head_width, ec="g", fc="g", alpha=.8,
                  zorder=81)
        # impact locations
        for loc in p_report.locations:
            loc_str, loc_shape = loc
            plt.fill(*loc_shape.exterior.xy, fc="b", ec="k", alpha=0.4, zorder=50)
            xc, yc = loc_shape.centroid.coords[0]
            plt.text(xc, yc, f"{loc_str}", horizontalalignment="center", verticalalignment="center", zorder=85)

    # common impact point and normals
    imp_point = collision_report.impact_point.coords[0]
    plt.plot(*imp_point, "o", zorder=79)
    n_color = "b"
    n = collision_report.impact_normal
    plt.arrow(imp_point[0], imp_point[1], n[0], n[1], ec=n_color, fc=n_color, alpha=.9, zorder=70)

    fig.set_tight_layout(True)
    plt.axis('equal')
    return
