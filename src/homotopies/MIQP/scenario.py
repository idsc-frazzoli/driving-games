import numpy as np
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.scenarios import load_commonroad_scenario
from dg_commons import PlayerName
from homotopies.MIQP.utils.prediction import predict, traj_from_commonroad
from homotopies.MIQP.utils.intersects import find_intersects


# scenario settings
def get_simple_scenario(n_player):
    player1 = PlayerName('p1')
    player2 = PlayerName('p2')
    player3 = PlayerName('p3')

    state1 = VehicleState(x=15, y=0, theta=np.pi / 2, vx=4.5, delta=0)
    state2 = VehicleState(x=-5, y=20, theta=np.pi / 3, vx=3.5, delta=0)
    state3 = VehicleState(x=-8, y=30, theta=0, vx=1, delta=0)

    if n_player == 2:
        obs = {player1: state1, player2: state2}
    else:
        obs = {player1: state1, player2: state2, player3: state3}

    trajs = predict(obs)

    intersects = find_intersects(trajs)

    if n_player == 2:
        x0 = np.array([0, state1.vx, 0, state2.vx])
    else:
        x0 = np.array([0, state1.vx, 0, state2.vx, 0, state3.vx])

    return trajs, intersects, x0, None


def get_commonroad_scenario():
    player1 = PlayerName('p1')
    player2 = PlayerName('p2')
    player3 = PlayerName('p3')

    state1 = VehicleState(x=-14, y=-70, theta=np.deg2rad(64), vx=3.5, delta=0)
    state2 = VehicleState(x=15, y=-52, theta=np.deg2rad(153), vx=4, delta=0)

    obs = {player1: state1, player2: state2}

    scenario_name = "USA_Lanker-2_10_T-1"
    scenario_dir = "/home/ysli/Desktop/SP/driving-games/scenarios/"  # path to the commonroad scenario folder
    scenario, _ = load_commonroad_scenario(scenario_name, scenario_dir)
    traj_p3 = traj_from_commonroad(scenario_name, scenario_dir, 2653, offset=(0, 0))
    vx_p3 = 3

    trajs = predict(obs)

    trajs[player3] = traj_p3

    intersects = find_intersects(trajs)

    x0 = np.array([0, state1.vx, 0, state2.vx, 0, vx_p3])

    return trajs, intersects, x0, scenario
