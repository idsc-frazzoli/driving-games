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
        # current_state.position[0] = current_state.position[0] - math.cos(current_state.orientation)*vehicle_params.l/2.0
        # current_state.position[1] = current_state.position[1] - math.sin(current_state.orientation)*vehicle_params.l/2.0
        beta = math.atan2((vehicle_params.l/2) * math.tan(current_state.steering_angle), vehicle_params.l)
        # current_state.velocity = current_state.velocity / math.cos(beta)**2
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

    initial_state = VehicleState(x=p[0], y=p[1], vx=10.0, theta=-0.02, delta=0)

    vehicle_type = VehicleType.FORD_ESCORT
    # vehicle_dynamics = VehicleDynamics.KS(vehicle_type)

    vehicle_params = vehicle_parameters[vehicle_type]

    traj_gen_params = traj_gen_params_from_cr(cr_vehicle_params=vehicle_params, is_ego=True)

    traj_gen_params.u_dst = frozenset([-0.5, 0.5])
    traj_gen_params.u_acc = frozenset([1.0, 2.0])
    traj_gen_params.s_final = -1
    traj_gen_params.n_factor = 1.2
    traj_gen_params.solve = False





    from decimal import Decimal as D
    traj_gen_params.dt = D("1.0")

    u_acc = frozenset([1.0, 3.0])  # todo: negative accelerations should also be possible
    u_dst = frozenset([-0.1, 0.1])  # todo: not only "concave starts", also convex should be possible
    # u_dst = frozenset([_ * 0.2 for _ in u_acc])

    # todo: generate less curves but "longer"?
    # traj_gen_params = TrajectoryGenParams(
    #     solve=True,
    #     s_final=-1,  # todo: add this to generator
    #     max_gen=100,
    #     dt=D("0.5"),  # keep at max 1 sec, increase k_maxgen in trajectrory_generator for having more generations
    #     u_acc=u_acc,
    #     u_dst=u_dst,
    #     v_max=15.0,
    #     v_min=0.0,
    #     st_max=0.5,
    #     dst_max=1.0,
    #     dt_samp=D("0.2"),
    #     dst_scale=False,
    #     n_factor=1.0,  # todo: investigate what this does
    #     vg=VehicleGeometry.default_car(),
    # )

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
    return trajectories


if __name__ == "__main__":
    trajectories = generate_trajectories()
    count = 0
    for traj in trajectories:
        feas = check_feasibility(traj)
        if feas:
            count = count+1
        print(feas)

    print(count)
