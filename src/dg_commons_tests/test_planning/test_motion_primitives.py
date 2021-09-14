from decimal import Decimal

import matplotlib.pyplot as plt

from dg_commons.dynamics import BicycleDynamics
from dg_commons.planning.motion_primitives import MotionPrimitivesGenerator, MPGParam
from sim.models.vehicle_structures import VehicleGeometry
from sim.models.vehicle_utils import VehicleParameters
from sim.simulator_visualisation import plot_trajectories


def test_generate_motion_primitives():
    vp = VehicleParameters.default_car()
    vg = VehicleGeometry.default_car()

    params = MPGParam(dt=Decimal(".2"),
                      n_steps=3,
                      velocity=(0, 50, 3),
                      steering=(-vp.delta_max, vp.delta_max, 3))
    vehicle = BicycleDynamics(vg=vg, vp=vp)
    mpg = MotionPrimitivesGenerator(mpg_param=params,
                                    vehicle_dynamics=vehicle.successor,
                                    vehicle_params=vp)

    traject = mpg.generate()
    # viz
    fig = plt.figure(figsize=(10,7),dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect("equal")
    traj_lines, traj_points = plot_trajectories(ax=ax, trajectories=list(traject))
    # set the limits
    ax.set_xlim([-1, 15])
    ax.set_ylim([-10, 10])
    #ax.autoscale(True)
    # plt.draw()
    plt.savefig("out/mpg_debug.png")
