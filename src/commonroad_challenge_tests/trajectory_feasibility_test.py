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
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.scenarios import load_commonroad_scenario
from dg_commons.sim.scenarios.agent_from_commonroad import dglane_from_position
from dg_commons_dev.utils import get_project_root_dir
from trajectory_games import TrajectoryGenerator, TrajectoryGenerationVisualization, TrajectoryGenParams
from commonroad.common.solution import CommonRoadSolutionReader
from decimal import Decimal as D


def check_feasibility(traj: Trajectory) -> bool:
    dt = traj.timestamps[1] - traj.timestamps[0]
    vehicle_type = VehicleType.FORD_ESCORT
    vehicle_dynamics = VehicleDynamics.KS(vehicle_type)

    # convert trajectory to commonroad trajectory
    cr_traj_states = []
    values = traj.values
    for i, state in enumerate(values):
        current_state = convert_to_cr_state(state)
        # current_state.position[0] = current_state.position[0] - math.cos(current_state.orientation)*vehicle_params.l/2.0
        # current_state.position[1] = current_state.position[1] - math.sin(current_state.orientation)*vehicle_params.l/2.0
        # current_state.velocity = current_state.velocity / math.cos(beta)**2
        current_state.time_step = i
        cr_traj_states.append(current_state)

    cr_traj = Trajectory(initial_time_step=0, state_list=cr_traj_states)
    # check feasibility of planned trajectory for the given vehicle model
    feasible, reconstructed_inputs = feasibility_checker.trajectory_feasibility(cr_traj, vehicle_dynamics, dt)
    # print('The planned trajectory is feasible: %s' % feasible)
    return feasible


def get_traj_from_solution(filename: str):
    solution = CommonRoadSolutionReader.open(filename)
    sol = solution.planning_problem_solutions[0]
    traj = sol.trajectory
    return traj


def generate_trajectories(show_plot=False):
    SCENARIOS_DIR = os.path.join(get_project_root_dir(), "scenarios")
    scenario, _ = load_commonroad_scenario("DEU_Ffb-1_7_T-1", SCENARIOS_DIR)
    matplotlib.use("TkAgg")

    p = np.array([42.0, 0.0])
    dglane = dglane_from_position(p, scenario.lanelet_network, succ_lane_selection=2)

    ref_lane_goals = [RefLaneGoal(ref_lane=dglane, goal_progress=1000.0)]

    initial_state = VehicleState(x=p[0], y=p[1], vx=30.0, theta=-0.02, delta=0)

    # import vehicle parameters from Commonroad
    vehicle_type = VehicleType.FORD_ESCORT
    vehicle_params = vehicle_parameters[vehicle_type]
    # generate trajectory generator parameters
    traj_gen_params = traj_gen_params_from_cr(cr_vehicle_params=vehicle_params, is_ego=True)

    # modify some parameters for testing
    traj_gen_params.u_dst = frozenset([-0.2, 0, 0.2])
    traj_gen_params.u_acc = frozenset([-2.0, 0.0, 2.0])
    traj_gen_params.s_final = -1
    # traj_gen_params.n_factor = 1.2
    traj_gen_params.dt = D("0.5")

    generator = TrajectoryGenerator(params=traj_gen_params, ref_lane_goals=ref_lane_goals)

    traj_graph = generator.get_actions(state=initial_state, return_graphs=True)
    graph = list(traj_graph)[0]
    trajectories = graph.get_all_trajectories()
    # trajectories = graph.get_all_transitions()

    show_plot = True
    if show_plot:
        viz = TrajectoryGenerationVisualization(scenario=scenario, trajectories=trajectories)

        dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "out")
        filename = os.path.join(dir_path, "trajectory_generation_test_0.8.png")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        viz.plot(show_plot=True, draw_labels=True, action_color="red", filename=filename)

    # braking test
    # state_0 = VehicleState(x=20, y=20, theta=0.2, vx=20.0, delta=0.0)
    # # emerg_traj = emergency_braking_trajectory(state=state_0, max_long_acc=vehicle_params.longitudinal.a_max, params=traj_gen_params, t_final=5.0)
    # break_traj = braking_trajectory(state=state_0, long_dec=5.0, max_long_acc=vehicle_params.longitudinal.a_max,
    #                                 params=traj_gen_params, t_final=5.0)
    # print(break_traj)
    # print("Emergency trajectory is feasible: ")
    # print(check_feasibility(break_traj))
    # print("END")
    # # exit()
    return trajectories


if __name__ == "__main__":
    filename = "/media/leon/Extreme SSD1/MT/experiments/09-05-22-180842/DEU_Dresden-3_19_I-1/with_ego/solution/solution_DEU_Dresden-3_19_I-1.xml"
    traj = get_traj_from_solution(filename)
    vehicle_type = VehicleType.FORD_ESCORT
    vehicle_dynamics = VehicleDynamics.KS(vehicle_type)

    # vehicle_params = vehicle_parameters[vehicle_type]
    dt = 0.1
    feasible, reconstructed_inputs = feasibility_checker.trajectory_feasibility(traj, vehicle_dynamics, dt)
    print(feasible)
    exit()
    trajectories = generate_trajectories()
    count = 0
    count_false = 0
    for traj in trajectories:
        feas = check_feasibility(traj)
        if feas:
            count = count + 1
        else:
            count_false += 1

    print("Feasible: ")
    print(count)
    print("Not feasible: ")
    print(count_false)
