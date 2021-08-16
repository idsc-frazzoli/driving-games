# import functions to read xml file and visualize commonroad objects

import matplotlib.pyplot as plt
from commonroad.visualization.mp_renderer import MPRenderer

# from commonroad_route_planner.route_planner import RoutePlanner
from sim.scenarios import load_commonroad_scenario
from sim.scenarios.agent_from_commonroad import npAgent_from_dynamic_obstacle


def test_load_commonroad():
    # generate path of the file to be opened
    scenario = "USA_Lanker-1_1_T-1.xml"

    scenario, planning_problem_set = load_commonroad_scenario(scenario)
    # print(scenario)
    # plt.figure()
    # draw_object(scenario)
    # draw_object(planning_problem_set)
    # plt.gca().set_aspect('equal')
    # plt.show()
    # lanelet_id = dyn_obs.initial_center_lanelet_ids
    # Lanelet.all_lanelets_by_merging_successors_from_lanelet(lanelet_id,network=scenario.lanelet_network)

    rnd = MPRenderer()
    # axis = rnd.ax
    # scenario.draw(rnd)
    # planning_problem_set.draw(rnd)
    for dyn_obs in scenario.dynamic_obstacles:
        dyn_obs.draw(rnd)
    scenario.lanelet_network.draw(rnd, draw_params={"traffic_light": {
        "draw_traffic_lights": False}})
    rnd.render()
    plt.savefig("debug.png", dpi=300)

    # write_default_params("../../sim_tests/scenarios_tests/default_params.json")


def test_npAgent_from_dynamic_obstacle():
    scenario = "USA_Lanker-1_1_T-1.xml"
    scenario, planning_problem_set = load_commonroad_scenario(scenario)
    dyn_obs = scenario.dynamic_obstacles[2]

    agent, model = npAgent_from_dynamic_obstacle(dyn_obs)

    print(agent, model)
