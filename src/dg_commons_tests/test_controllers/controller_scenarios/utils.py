import math
import numpy as np
from sim.scenarios.utils import Scenario
from commonroad.visualization.util import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction, Trajectory
from sim.scenarios.agent_from_commonroad import *
import matplotlib.pyplot as plt


def race_track_generate_dyn_obs(scenario: Scenario, starting_position: float = 0, length_perc: float = 100):
    lanelets = scenario.lanelet_network.lanelets
    states = []
    n_states = []

    for lanelet in lanelets:
        center_vertices = lanelet.center_vertices.tolist()
        n_states.append(len(center_vertices))
        for i, center_vertice in enumerate(center_vertices):
            n = len(center_vertices)
            if i == n-1:
                helper = center_vertices[i-1]
                orientation = math.atan2(center_vertice[1] - helper[1], center_vertice[0] - helper[0])
            else:
                helper = center_vertices[i + 1]
                orientation = math.atan2(helper[1] - center_vertice[1], helper[0] - center_vertice[0])
            state = State(position=np.array(center_vertice), orientation=orientation, time_step=0, velocity=0.0)
            states.append(state)

    further_separation = 2
    n_lanelet = int(len(lanelets)*starting_position/100)
    n_center = sum(n_states[:n_lanelet])-1+further_separation

    dec_states = states[n_center:]
    length = int(len(states) * length_perc / 100)
    n_add_states = length - len(dec_states)
    if n_add_states <= 0:
        dec_states = states[n_center:(n_center+length)]
    else:
        dec_states = dec_states + states[:n_add_states]

    '''x, y = [q.position[0] for q in dec_states], [q.position[1] for q in dec_states]
    plt.scatter([x[0], x[-1]], [y[0], y[-1]])
    plt.plot(x, y)
    plt.savefig("Test")'''

    initial_state = dec_states[0]
    trajectory = Trajectory(0, dec_states)

    dyn_obs = DynamicObstacle(obstacle_id=0, obstacle_type=ObstacleType.CAR,
                              obstacle_shape=Rectangle(1, 1), initial_state=initial_state,
                              prediction=TrajectoryPrediction(trajectory, shape=Rectangle(1, 1)))

    return [dyn_obs]
