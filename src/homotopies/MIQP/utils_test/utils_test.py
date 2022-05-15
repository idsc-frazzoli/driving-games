import matplotlib.pyplot as plt
import matplotlib
from reprep import Report
import numpy as np
from dg_commons import PlayerName, DgSampledSequence
from dg_commons.sim.models.vehicle import VehicleState
from homotopies.MIQP.utils.prediction import predict, traj2lane
from homotopies.MIQP.utils.intersects import find_intersects
from homotopies.MIQP.utils.visualization import *
from homotopies.MIQP.utils.report import generate_report_all, generate_report_3d_boxes


# state1 = VehicleState(x=15, y=0, theta=np.pi / 2, vx=5, delta=0.03)
state1 = VehicleState(x=15, y=0, theta=np.pi / 2, vx=4.5, delta=0)
state2 = VehicleState(x=-5, y=20, theta=np.pi/3, vx=3.5, delta=0)
# state2 = VehicleState(x=-5, y=0, theta=np.pi / 4, vx=5, delta=-0.03)
state3 = VehicleState(x=-8, y=30, theta=0, vx=1, delta=0)

player1 = PlayerName('p1')
player2 = PlayerName('p2')
# player3 = None
player3 = PlayerName('p3')
if player3 is None:
    obs = {player1: state1, player2: state2}
else:
    obs = {player1: state1, player2: state2, player3: state3}

trajs = predict(obs)
lane = traj2lane(trajs[player1])
intersects = find_intersects(trajs)

# # visualization
colors = {player1: 'blue', player2: 'green', player3: 'black'}
if player3 is not None:
    plotnum = 3
else:
    plotnum = 3

matplotlib.use('TkAgg')
fig, axs = plt.subplots(plotnum, 2)
gs = axs[0, 0].get_gridspec()
# remove the underlying axes
for ax in axs[:, 0]:
    ax.remove()
ax_traj = fig.add_subplot(gs[:, 0])

visualize_trajs_all(trajs, intersects, ax_traj, colors)
visualize_car(trajs[player1].at(0), ax_traj)
if player2 in intersects[player1].keys():
    visualize_box_2d(trajs, intersects, player1, player2, axs[0, 1])
if player3 is not None:
    if player3 in intersects[player1].keys():
        visualize_box_2d(trajs, intersects, player1, player3, axs[1, 1])
    if player3 in intersects[player2].keys():
        visualize_box_2d(trajs, intersects, player2, player3, axs[2, 1])

ax_3d = plt.figure().add_subplot(projection='3d')
if player3 is not None \
        and player3 in intersects[player1].keys() \
        and player3 in intersects[player2].keys() \
        and player2 in intersects[player1].keys():
    visualize_box_3d(trajs, intersects, player1, player2, player3, ax_3d)

plt.show()

# create report
# r = generate_report_all(trajs, intersects, colors)
# r.to_html('utils_report')
