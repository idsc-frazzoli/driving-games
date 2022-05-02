import os

import matplotlib
import matplotlib.pyplot as plt

from commonroad.common.file_reader import CommonRoadFileReader


# draw_object
from commonroad_challenge.maneuver_automaton.uility import plot_primitives
from dg_commons.sim.scenarios import load_commonroad_scenario
from dg_commons_dev.utils import get_project_root_dir
from maneuver_automaton import ManeuverAutomaton
from motion_primitive_generator import MotionPrimitiveGenerator as MPG
# from SMP.motion_planner.motion_planner import MotionPlanner
# from SMP.motion_planner.utility import plot_primitives, display_steps
import numpy as np

if __name__ == "__main__":
    SCENARIOS_DIR = os.path.join(get_project_root_dir(), "scenarios")
    scenario, planning_problem_set = load_commonroad_scenario("DEU_Ffb-1_7_T-1", SCENARIOS_DIR)
    matplotlib.use("TkAgg")

    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    name_file_motion_primitives = 'V_0.0_20.0_Vstep_1.0_SA_-0.91_0.91_SAstep_0.23_T_0.5_Model_FORD_ESCORT.xml'
    # automaton = ManeuverAutomaton.generate_automaton(name_file_motion_primitives)

    # all_primitives = automaton.list_primitives
    #
    # connectivity = automaton.examine_connectivity(verbose=True)
    # a=10
    # plot_primitives(automaton.list_primitives)

    # specify path to configuration file
    path_file_config = "./generator_config.yaml"

    # load configuration file
    MPG.load_configuration(path_file_config)

    list_motion_primitives = MPG.generate_motion_primitives()


    fig = plt.figure(figsize=(12, 3))
    ax = fig.gca()

    for traj in list_motion_primitives:
        list_x = [state.position[0] for state in traj.state_list]
        list_y = [state.position[1] for state in traj.state_list]
        plt.plot(list_x, list_y)

    ax.set_xticks(np.arange(-5, 20, 0.5))
    ax.set_yticks(np.arange(-5, 5., 0.5))
    plt.axis('equal')
    plt.grid(alpha=0.5)
    plt.xlim((-1, 11))
    plt.ylim((-1, 2))
    plt.show()

    list_motion_primitives_mirrored = MPG.create_mirrored_primitives(list_motion_primitives)
    print("Total number of primitives (mirrored included): ", len(list_motion_primitives_mirrored))

    # branching_factor_average = MPG.compute_branching_factor(list_motion_primitives_mirrored)
    # print("Average branching factor of primitives: ", branching_factor_average)

    all_trajs = MPG.generate_all_trajectories(list_motion_primitives_mirrored)
    MPG.generate_sample_trajectories(list_motion_primitives_mirrored)







