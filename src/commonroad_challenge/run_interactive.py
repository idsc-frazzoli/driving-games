import os
import random
import sys
from datetime import datetime

from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.common.solution import VehicleType, VehicleModel, CostFunction
from commonroad.scenario.scenario import Tag
from simulation.simulations import simulate_without_ego, simulate_with_planner
from simulation.utility import save_solution

from planning import motion_planner_from_trajectory
from utils import find_all_scenarios

sys.path.append(os.path.join(os.getcwd(), "../"))


def run_all_scenarios(with_ego: bool, without_ego: bool, scenarios_dir: str, output_dir: str):
    all_paths = find_all_scenarios(scenarios_dir)
    random.shuffle(all_paths)

    for scenario_path in all_paths:
        scenario_path = os.path.join(scenarios_dir, scenario_path)
        run_scenario(with_ego, without_ego, scenario_path, output_dir)

    return


def run_scenario(with_ego: bool, without_ego: bool, scenario_path: str, output_dir: str):
    scenario_name = os.path.basename(os.path.normpath(scenario_path))
    output_dir = os.path.join(output_dir, scenario_name)

    if with_ego:
        out_dir = os.path.join(output_dir, "with_ego")
        simulate_with_ego(scenario_path=scenario_path, output_dir=out_dir, create_video=True, store_solution=True)
    if without_ego:
        out_dir = os.path.join(output_dir, "no_ego")
        simulate_no_ego(scenario_path=scenario_path, output_dir=out_dir, create_video=True)
    return


def simulate_no_ego(scenario_path: str, output_dir: str, create_video: bool = False):
    scenario_without_ego, pps = simulate_without_ego(interactive_scenario_path=scenario_path,
                                                     output_folder_path=output_dir,
                                                     create_video=create_video)
    # write simulated scenario to CommonRoad xml file
    simulated_scenarios_path = os.path.join(output_dir, "simulated_scenarios")
    if not os.path.exists(simulated_scenarios_path):
        os.makedirs(simulated_scenarios_path)
    fw = CommonRoadFileWriter(scenario_without_ego, pps, author, affiliation, source, tags)
    fw.write_to_file(
        os.path.join(simulated_scenarios_path, os.path.basename(os.path.normpath(scenario_path)) + "_no_ego.xml"),
        OverwriteExistingFile.ALWAYS)


def simulate_with_ego(scenario_path: str, output_dir: str, create_video: bool = False, store_solution: bool = False):
    # video_path = os.path.join(output_dir, "videos")
    # if not os.path.exists(video_path):
    #     os.makedirs(video_path)

    scenario_with_planner, pps, ego_vehicles_planner = simulate_with_planner(interactive_scenario_path=scenario_path,
                                                                             motion_planner=motion_planner_from_trajectory,
                                                                             output_folder_path=output_dir,
                                                                             create_video=create_video)

    # matplotlib.use("TkAgg")
    # visualize_scenario_with_trajectory(scenario_with_planner, pps, ego_vehicles_planner)

    if scenario_with_planner:
        # write simulated scenario to CommonRoad xml file
        simulated_scenarios_path = os.path.join(output_dir, "simulated_scenarios")
        if not os.path.exists(simulated_scenarios_path):
            os.makedirs(simulated_scenarios_path)
        fw = CommonRoadFileWriter(scenario_with_planner, pps, author, affiliation, source, tags)
        fw.write_to_file(
            os.path.join(simulated_scenarios_path, os.path.basename(os.path.normpath(scenario_path)) + "_planner.xml"),
            OverwriteExistingFile.ALWAYS)

        if store_solution:
            # save the planned trajectory to solution file
            solution_path = os.path.join(output_dir, "solution")
            if not os.path.exists(solution_path):
                os.makedirs(solution_path)
            save_solution(scenario_with_planner, pps, ego_vehicles_planner, vehicle_type, vehicle_model, cost_function,
                          solution_path, overwrite=True)


if __name__ == "__main__":
    # directory where scenarios are stored
    scenarios_dir = "/media/leon/Extreme SSD1/MT/scenarios_phase_1"
    # directory to store outputs
    output_dir = "/media/leon/Extreme SSD1/MT/experiments/"
    output_dir = os.path.join(output_dir, datetime.now().strftime("%d-%m-%y-%H%M%S"))

    # attributes for saving the simulated scenarios
    author = 'Leon Zueger'
    affiliation = 'ETH Zurich, Switzerland'
    source = ''
    tags = {Tag.URBAN}

    vehicle_type = VehicleType.FORD_ESCORT
    vehicle_model = VehicleModel.KS
    cost_function = CostFunction.TR1

    # run all scenario in scenarios_dir
    run_all_scenarios(with_ego=True, without_ego=True, scenarios_dir=scenarios_dir, output_dir=output_dir)

    # run a single scenario at scenario_path
    # deu_dresden_3_12_path = "/media/leon/Extreme SSD1/MT/scenarios_phase_1/DEU_Dresden-3_12_I-1"
    # run_scenario(with_ego=True, without_ego=True, scenario_path=deu_dresden_3_12_path, output_dir=output_dir)
