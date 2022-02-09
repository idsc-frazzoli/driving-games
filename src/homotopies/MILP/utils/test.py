import matplotlib.pyplot as plt
import numpy as np
from dg_commons import PlayerName, DgSampledSequence
from dg_commons.sim.models.vehicle import VehicleState
from homotopies.MILP.utils.prediction import predict
from homotopies.MILP.utils.intersects import find_intersects, traj2path, get_s_max
from homotopies.MILP.utils.visualization import *

state1 = VehicleState(x=15, y=0, theta=np.pi / 2, vx=5, delta=0.03)
state2 = VehicleState(x=-5, y=0, theta=np.pi / 4, vx=5, delta=-0.03)
state3 = VehicleState(x=-8, y=15, theta=0, vx=4, delta=0)

player1 = PlayerName('p1')
player2 = PlayerName('p2')
# player3 = None
player3 = PlayerName('p3')
if player3 is None:
    obs = {player1: state1, player2: state2}
else:
    obs = {player1: state1, player2: state2, player3: state3}

trajs = predict(obs)
intersects = find_intersects(trajs)

if player3 is not None:
    rownum = 3
    color={player1:'blue', player2:'green', player3: 'black'}
else:
    rownum = 1
    color = {player1: 'blue', player2: 'green'}

fig, axs = plt.subplots(rownum, 2)
gs = axs[0, 0].get_gridspec()
# remove the underlying axes
for ax in axs[:, 0]:
    ax.remove()
ax_traj = fig.add_subplot(gs[:, 0])

for player in trajs.keys():
    visualize_traj(trajs[player], player, ax_traj, color[player])

if player2 in intersects[player1].keys():
    visualize_intersect_all(trajs, intersects, player1, player2, ax_traj, axs[0, 1])
if player3 is not None:
    if player3 in intersects[player1].keys():
        visualize_intersect_all(trajs, intersects, player1, player3, ax_traj, axs[1, 1])
    if player3 in intersects[player2].keys():
        visualize_intersect_all(trajs, intersects, player2, player3, ax_traj, axs[2, 1])

ax_3d = plt.figure().add_subplot(projection='3d')
if player3 is not None and player3 in intersects[player1].keys() and player3 in intersects[player2].keys():
    visualize_box_3d(trajs, intersects, player1, player2, player3, ax_3d)

plt.show()
