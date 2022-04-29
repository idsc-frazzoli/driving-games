import math
import os

import commonroad_dc.feasibility.feasibility_checker as feasibility_checker
import matplotlib
import numpy as np
from commonroad_dc.feasibility.vehicle_dynamics import VehicleDynamics
from commonroad.common.solution import VehicleType, vehicle_parameters
from commonroad_challenge.utils import traj_gen_params_from_cr, convert_to_cr_state
# set time step as scenario time step
from dg_commons.planning import RefLaneGoal, Trajectory
from commonroad.scenario.trajectory import Trajectory
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.scenarios import load_commonroad_scenario
from dg_commons.sim.scenarios.agent_from_commonroad import dglane_from_position
from dg_commons_dev.utils import get_project_root_dir
from trajectory_games import TrajectoryGenerator, TrajectoryGenerationVisualization


def check_feasibility(traj: Trajectory) -> bool:
    dt = traj.timestamps[1] - traj.timestamps[0]
    vehicle_type = VehicleType.FORD_ESCORT
    vehicle_dynamics = VehicleDynamics.KS(vehicle_type)

    vehicle_params = vehicle_parameters[vehicle_type]

    # convert trajectory to commonroad trajectory
    cr_traj_states = []
    values = traj.values
    for i, state in enumerate(values):
        current_state = convert_to_cr_state(state)
        current_state.position[0] = current_state.position[0] - math.cos(current_state.orientation)*vehicle_params.l/2.0
        current_state.position[1] = current_state.position[1] - math.sin(current_state.orientation)*vehicle_params.l/2.0
        current_state.time_step = i
        cr_traj_states.append(current_state)

    cr_traj = Trajectory(initial_time_step=0, state_list=cr_traj_states)
    # check feasibility of planned trajectory for the given vehicle model
    feasible, reconstructed_inputs = feasibility_checker.trajectory_feasibility(cr_traj, vehicle_dynamics, dt)
    # print('The planned trajectory is feasible: %s' % feasible)
    return feasible


def generate_trajectories(show_plot=False):
    SCENARIOS_DIR = os.path.join(get_project_root_dir(), "scenarios")
    scenario, _ = load_commonroad_scenario("DEU_Ffb-1_7_T-1", SCENARIOS_DIR)
    matplotlib.use("TkAgg")

    p = np.array([42.0, 0.0])
    dglane = dglane_from_position(p, scenario.lanelet_network, succ_lane_selection=2)

    ref_lane_goals = [RefLaneGoal(ref_lane=dglane, goal_progress=1000)]

    initial_state = VehicleState(x=p[0], y=p[1], vx=0, theta=-0.02, delta=0)

    vehicle_type = VehicleType.FORD_ESCORT
    # vehicle_dynamics = VehicleDynamics.KS(vehicle_type)

    vehicle_params = vehicle_parameters[vehicle_type]

    traj_gen_params = traj_gen_params_from_cr(cr_vehicle_params=vehicle_params)
    generator = TrajectoryGenerator(params=traj_gen_params, ref_lane_goals=ref_lane_goals)

    traj_graph = generator.get_actions(state=initial_state, return_graphs=True)
    graph = list(traj_graph)[0]
    trajectories = graph.get_all_trajectories()

    if show_plot:
        viz = TrajectoryGenerationVisualization(scenario=scenario, trajectories=trajectories)

        dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "out")
        filename = os.path.join(dir_path, "trajectory_generation_test_0.8.png")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        viz.plot(show_plot=True, draw_labels=True, action_color="red", filename=filename)
    return trajectories


if __name__ == "__main__":
    trajectories = generate_trajectories()

    for traj in trajectories:
        print(check_feasibility(traj))
