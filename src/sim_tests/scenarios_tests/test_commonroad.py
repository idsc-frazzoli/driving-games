# import functions to read xml file and visualize commonroad objects
import json

from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.param_server import write_default_params

from sim.scenarios import load_commonroad_scenario


def test_load_commonroad():
    # generate path of the file to be opened
    scenario = "USA_Lanker-1_1_T-1.xml"

    scenario, planning_problem_set = load_commonroad_scenario(scenario)
    print(scenario)
    # plt.figure()
    # draw_object(scenario)
    # draw_object(planning_problem_set)
    # plt.gca().set_aspect('equal')
    # plt.show()

    rnd = MPRenderer()
    axis = rnd.ax
    # scenario.draw(rnd)
    # planning_problem_set.draw(rnd)
    scenario.lanelet_network.draw(rnd, draw_params={"traffic_light": {
        "draw_traffic_lights": False}})
    # rnd.render(show=True)

    write_default_params("../../sim_tests/scenarios_tests/default_params.json")
