import copy
import os, sys

import numpy as np
from math import sin, cos

from commonroad.scenario.trajectory import State

sys.path.append(os.path.join(os.getcwd(), "../"))

from simulation.simulations import simulate_without_ego, simulate_with_solution, simulate_with_planner
from simulation.utility import visualize_scenario_with_trajectory, save_solution
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.common.solution import CommonRoadSolutionReader, VehicleType, VehicleModel, CostFunction
from commonroad.scenario.scenario import Tag


def interacting_agents(scenario: Scenario, ego_state: State, threshold: float = 20) -> List[int]:
    interacting_dynamic_obstacles = []
    for dyn_obs in scenario.dynamic_obstacles:
        dpos = ego_state.position - dyn_obs.initial_state.position
        if np.linalg.norm(dpos) < threshold:
            interacting_dynamic_obstacles.append(dyn_obs.obstacle_id)
    return interacting_dynamic_obstacles

def find_all_scenarios(scnearios_folder: str):
    subfolders = [x[0] for x in os.walk(scnearios_folder)]
    interactive_scenarios = [os.path.split(path)[1] for path in subfolders[1:]]
    return interactive_scenarios


def dummy_motion_planner(state: State) -> State:
    state.steering_angle = 0.0
    a = -4.0
    dt = 0.1
    if state.velocity > 0:
        v = state.velocity
        x, y = state.position
        o = state.orientation

        state.position = np.array([x + v * cos(o) * dt, y + v * sin(o) * dt])
        # next_state.velocity += a * dt
    return state


def generate_all_videos():
    all_paths = find_all_scenarios("/media/leon/Extreme SSD/MT/scenarios_phase_1")

    for path in all_paths:
        # simulate_no_ego(scenario_name=path, create_video=True)
        simulate_with_ego(scenario_name=path, create_video=True)


def simulate_no_ego(scenario_name: str, create_video: bool = False):
    path_scenario = os.path.join(folder_scenarios, scenario_name)
    scenario_without_ego, pps = simulate_without_ego(interactive_scenario_path=path_scenario,
                                                     output_folder_path=path_video,
                                                     create_video=create_video)
    # write simulated scenario to CommonRoad xml file
    fw = CommonRoadFileWriter(scenario_without_ego, pps, author, affiliation, source, tags)
    fw.write_to_file(os.path.join(path_scenarios_simulated, scenario_name + "_no_ego.xml"),
                     OverwriteExistingFile.ALWAYS)


def simulate_with_ego(scenario_name: str, create_video: bool = False, store_solution: bool = False):
    path_scenario = os.path.join(folder_scenarios, scenario_name)
    scenario_with_planner, pps, ego_vehicles_planner = simulate_with_planner(interactive_scenario_path=path_scenario,
                                                                             motion_planner=dummy_motion_planner,
                                                                             output_folder_path=path_video,
                                                                             create_video=create_video)
    if scenario_with_planner:
        # write simulated scenario to file
        fw = CommonRoadFileWriter(scenario_with_planner, pps, author, affiliation, source, tags)
        fw.write_to_file(os.path.join(path_scenarios_simulated, scenario_name + "_planner.xml"),
                         OverwriteExistingFile.ALWAYS)

        if store_solution:
            # save the planned trajectory to solution file
            save_solution(scenario_with_planner, pps, ego_vehicles_planner, vehicle_type, vehicle_model, cost_function,
                          path_solutions, overwrite=True)


if __name__ == "__main__":
    # path where scenarios are stored
    folder_scenarios = "/media/leon/Extreme SSD/MT/scenarios_phase_1"
    # path where solutions are saved
    path_solutions = "/media/leon/Extreme SSD/MT/outputs_dev/solutions"
    # path to save output video
    path_video = "/media/leon/Extreme SSD/MT/outputs_dev/videos"
    # path to save simulated scenarios
    path_scenarios_simulated = "/media/leon/Extreme SSD/MT/outputs_dev/simulated_scenarios"

    # attributes for saving the simulated scenarios
    author = 'Leon Zueger'
    affiliation = 'ETH Zurich, Switzerland'
    source = ''
    tags = {Tag.URBAN}

    vehicle_type = VehicleType.FORD_ESCORT
    vehicle_model = VehicleModel.KS
    cost_function = CostFunction.TR1

    generate_all_videos()

##################### SIMULATION WITH SOLUTION TRAJECTORY #################################
# we don't have solution file yet
# name_solution = "solution_KS1:TR1:DEU_Frankfurt-34_11_I-1:2020a"
# solution = CommonRoadSolutionReader.open(os.path.join(path_solutions, name_solution + ".xml"))
# # run simulation, a video of the simulation is stored in the end
# scenario_with_solution, pps, ego_vehicles_solution = simulate_with_solution(interactive_scenario_path=path_scenario,
#                                                                             output_folder_path=path_video,
#                                                                             solution=solution,
#                                                                             create_video=True)
# # write simulated scenario to CommonRoad xml file
# if scenario_with_solution:
#     # write simulated scenario to file
#     fw = CommonRoadFileWriter(scenario_with_solution, pps, author, affiliation, source, tags)
#     fw.write_to_file(os.path.join(path_scenarios_simulated, name_scenario + "_solution.xml"), OverwriteExistingFile.ALWAYS)


##################### END ####################################
