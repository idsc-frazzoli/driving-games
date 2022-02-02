from commonroad.planning.goal import GoalRegion
from commonroad.scenario.scenario import Scenario
from commonroad.common.util import Interval, AngleInterval
from commonroad.scenario.trajectory import State
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.param_server import write_default_params, ParamServer
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.geometry.shape import Rectangle
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
import numpy as np
from numpy import deg2rad
from typing import List
from sim.scenarios.utils import load_commonroad_scenario
import matplotlib.pyplot as plt


def planning_problem_4way_intersection(assignments: List[tuple]) -> None:
    """
    Generate initial states and goal regions for up to 4 agents on a 4-way intersection.
    We index the 4 ways by starting counting from the southern entry/exit, i.e. in negative y-direction.
    Then count counter-clockwise.

    :param assignments: list of tuples where first element of each tuple is the entry and the second element is the exit
                        from the 4-way intersection.
    :return: planning problem set
    """
    scenario_name = "ZAM_4_way_intersection-1_1_T-1.xml"

    scenario, planning_problem_set = load_commonroad_scenario(scenario_name=scenario_name)

    # initial states for each entry
    initial_state_0 = State(position=np.array([26.5, -20.0]), time_step=0,
                            velocity=5, yaw_rate=0, orientation=deg2rad(90), slip_angle=0)
    initial_state_1 = State(position=np.array([48.0, 3.0]), time_step=0,
                            velocity=5, yaw_rate=0, orientation=deg2rad(180), slip_angle=0)
    initial_state_2 = State(position=np.array([23.5, 24.0]), time_step=0,
                            velocity=5, yaw_rate=0, orientation=deg2rad(270), slip_angle=0)
    initial_state_3 = State(position=np.array([2.0, 0.0]), time_step=0,
                            velocity=5, yaw_rate=0, orientation=deg2rad(0), slip_angle=0)

    initial_states = [initial_state_0, initial_state_1, initial_state_2, initial_state_3]

    # final spatial goal regions for each exit
    goal_shape_0 = Rectangle(length=2, width=3, center=np.array([23.5, -20.0]))
    goal_shape_1 = Rectangle(length=3, width=2, center=np.array([48.0, 0.0]))
    goal_shape_2 = Rectangle(length=2, width=3, center=np.array([26.5, 24.0]))
    goal_shape_3 = Rectangle(length=3, width=2, center=np.array([2.0, 3.0]))

    # final states for each exit
    goal_state_0 = State(position=goal_shape_0, time_step=Interval(15, 20), orientation=AngleInterval(0.1, 1),
                         velocity=Interval(20, 30.5))
    goal_state_1 = State(position=goal_shape_1, time_step=Interval(15, 20), orientation=AngleInterval(0.1, 1),
                         velocity=Interval(20, 30.5))
    goal_state_2 = State(position=goal_shape_2, time_step=Interval(15, 20), orientation=AngleInterval(0.1, 1),
                         velocity=Interval(20, 30.5))
    goal_state_3 = State(position=goal_shape_3, time_step=Interval(15, 20), orientation=AngleInterval(0.1, 1),
                         velocity=Interval(20, 30.5))

    # in any GoalRegion there could be multiple states
    goal_regions = [GoalRegion([goal_state_0]), GoalRegion([goal_state_1]),
                    GoalRegion([goal_state_2]), GoalRegion([goal_state_3])]

    planning_problems = []
    for idx, pair in enumerate(assignments):
        initial_state = initial_states[pair[0]]
        goal_region = goal_regions[pair[1]]
        planning_problems.append(PlanningProblem(planning_problem_id=idx, initial_state=initial_state,
                                                 goal_region=goal_region))

    planning_problem_set = PlanningProblemSet(planning_problem_list=planning_problems)

    fw = CommonRoadFileWriter(scenario, planning_problem_set, "Leon Zueger", "ETH Zurich")

    filename = "ZAM_4_way_intersection-1_1_T-1_problem_set.xml"
    fw.write_to_file(filename, OverwriteExistingFile.ALWAYS)

    return


def planning_problem_highway_merging(assignments: List[tuple]) -> None:
    """
    Generate initial states and goal regions on a highway merging scenario arbitrary number of lanes, for up to 3 agents.
    We index the lanes by starting counting the left lanelet, for a system where the lanelets are straight and
    directed toward the positive y-axis (north).
    The placement can also be specified, i.e. who is starting/ending in front and who behind.
    For now can't handle repeated placements.

    :param assignments: list of tuples where: (start lanelet, start placement, end lanelet, end placement)
    :return: planning problem set
    """
    scenario_name = "ZAM_2_lane_merging-1_1_T-1.xml"
    scenario_name_2 = "ZAM_3_lane_merging-1_1_T-1.xml"

    scenario, planning_problem_set = load_commonroad_scenario(scenario_name=scenario_name)

    lane_width = 3.0
    delta_length = 6.0

    # initial states for each entry
    initial_state_0 = State(position=np.array([0.0, 0.0]), time_step=0,
                            velocity=5, yaw_rate=0, orientation=deg2rad(90), slip_angle=0)

    # final spatial goal regions for each exit
    goal_shape_0 = Rectangle(length=2, width=3, center=np.array([0.0, 80.0]))

    # final states for each exit
    goal_state_0 = State(position=goal_shape_0, time_step=Interval(15, 20), orientation=AngleInterval(0.1, 1),
                         velocity=Interval(20, 30.5))

    initial_tuples = []
    goal_tuples = []
    for assi in assignments:
        initial_tuples.append((assi[0], assi[1]))
        goal_tuples.append((assi[2], assi[3]))

    planning_problems = []
    for idx, pair in enumerate(assignments):
        initial_tuple = (pair[0], pair[1])  # 0,1

        initial_state = initial_state_0.translate_rotate(
            translation=np.array([initial_tuple[0] * lane_width, initial_tuple[1] * delta_length]), angle=0)

        goal_tuple = (pair[2], pair[3])  # 0,1
        goal_state = goal_state_0.translate_rotate(
            translation=np.array([goal_tuple[0] * lane_width, goal_tuple[1] * delta_length]), angle=0)
        goal_region = GoalRegion([goal_state])
        planning_problems.append(PlanningProblem(planning_problem_id=idx, initial_state=initial_state,
                                                 goal_region=goal_region))

    planning_problem_set = PlanningProblemSet(planning_problem_list=planning_problems)

    fw = CommonRoadFileWriter(scenario, planning_problem_set, "Leon Zueger", "ETH Zurich")

    filename = "ZAM_2_lane_merging-1_1_T-1_problem_set.xml"
    filename_2 = "ZAM_3_lane_merging-1_1_T-1_problem_set.xml"
    fw.write_to_file(filename, OverwriteExistingFile.ALWAYS)

    return


def planning_problem_roundabout(assignments: List[tuple]) -> None:
    """
    Generate initial states and goal regions at a roundabout with 4 entries and 4 exits (South, East, North, South).
    We index the 4 ways by starting counting from the southern entry/exit, i.e. in negative y-direction.
    Then count counter-clockwise. Degrees are counted from the southern exit and increasing counter-clockwise.

    :param assignments: list of tuples where (0,1,"out") indicates that a problem starts at entry 0 and has
    goal at exit 1. (120, 2, "in") indicates that a problem starts inside the roundabout, at angle 120 degrees,
    and stops at exit 2.
    :return: planning problem set
    """
    scenario_name = "ZAM_roundabout-1_1_T-1.xml"

    scenario, planning_problem_set = load_commonroad_scenario(scenario_name=scenario_name)

    radius = 10.0

    # initial states for each entry
    initial_state_center = State(position=np.array([0.0, 10.0]), time_step=0,
                                 velocity=5, yaw_rate=0, orientation=deg2rad(0), slip_angle=0)
    initial_state_0 = State(position=np.array([1.5, -7]), time_step=0,
                            velocity=5, yaw_rate=0, orientation=deg2rad(90), slip_angle=0)
    initial_state_1 = State(position=np.array([17.5, 11.5]), time_step=0,
                            velocity=5, yaw_rate=0, orientation=deg2rad(180), slip_angle=0)
    initial_state_2 = State(position=np.array([-1.5, 28.0]), time_step=0,
                            velocity=5, yaw_rate=0, orientation=deg2rad(270), slip_angle=0)
    initial_state_3 = State(position=np.array([-18.0, 8.0]), time_step=0,
                            velocity=5, yaw_rate=0, orientation=deg2rad(0), slip_angle=0)

    initial_states = [initial_state_0, initial_state_1, initial_state_2, initial_state_3]

    # final spatial goal regions for each exit
    goal_shape_0 = Rectangle(length=2, width=3, center=np.array([-1.5, -7]))
    goal_shape_1 = Rectangle(length=3, width=2, center=np.array([17.5, 8.0]))
    goal_shape_2 = Rectangle(length=2, width=3, center=np.array([1.5, 28.0]))
    goal_shape_3 = Rectangle(length=3, width=2, center=np.array([-18.0, 11.0]))

    # final states for each exit
    goal_state_0 = State(position=goal_shape_0, time_step=Interval(15, 20), orientation=AngleInterval(0.1, 1),
                         velocity=Interval(20, 30.5))
    goal_state_1 = State(position=goal_shape_1, time_step=Interval(15, 20), orientation=AngleInterval(0.1, 1),
                         velocity=Interval(20, 30.5))
    goal_state_2 = State(position=goal_shape_2, time_step=Interval(15, 20), orientation=AngleInterval(0.1, 1),
                         velocity=Interval(20, 30.5))
    goal_state_3 = State(position=goal_shape_3, time_step=Interval(15, 20), orientation=AngleInterval(0.1, 1),
                         velocity=Interval(20, 30.5))

    # in any GoalRegion there could be multiple states
    goal_regions = [GoalRegion([goal_state_0]), GoalRegion([goal_state_1]),
                    GoalRegion([goal_state_2]), GoalRegion([goal_state_3])]

    planning_problems = []
    for idx, tup in enumerate(assignments):
        if tup[-1] == "in":
            # translation = np.array([np.cos(deg2rad(tup[0] - 90)), np.sin(deg2rad(tup[0] - 90))]) * radius

            initial_state = State(position=np.array(
                [np.cos(deg2rad(tup[0] - 90)), np.sin(deg2rad(tup[0] - 90))]) * radius + np.array([0, 10.0]),
                                  time_step=0, velocity=5, yaw_rate=0, orientation=deg2rad(tup[0]),
                                  slip_angle=0)
            print(initial_state)

            # print(translation)
            # initial_state = initial_state_center.translate_rotate(translation=translation, angle=0)
            # initial_state = initial_state.translate_rotate(translation=np.array([0.0,0.0]), angle = deg2rad(90))

            goal_region = goal_regions[tup[1]]
            planning_problems.append(PlanningProblem(planning_problem_id=idx, initial_state=initial_state,
                                                     goal_region=goal_region))

        elif tup[-1] == "out":
            initial_state = initial_states[tup[0]]
            goal_region = goal_regions[tup[1]]
            planning_problems.append(PlanningProblem(planning_problem_id=idx, initial_state=initial_state,
                                                     goal_region=goal_region))

    planning_problem_set = PlanningProblemSet(planning_problem_list=planning_problems)

    fw = CommonRoadFileWriter(scenario, planning_problem_set, "Leon Zueger", "ETH Zurich")

    filename = "ZAM_roundabout-1_1_T-1_problem_set.xml"
    fw.write_to_file(filename, OverwriteExistingFile.ALWAYS)

    return


def plot_start_goal(filename: str):
    for i in range(0, 40):
        if i % 10 == 0:
            plt.figure(figsize=(25, 10))
            rnd = MPRenderer()
            scenario.draw(rnd, draw_params={'time_begin': i})
            write_default_params("default_parameters.json")
            param_server = ParamServer().from_json("default_parameters.json")

            colors = ['r', 'b', 'y', 'g']
            for j in range(len(planning_problem_set.planning_problem_dict)):
                param_server["initial_state"]["state"]["kwargs"]["color"] = colors[j]
                param_server["initial_state"]["state"]["radius"] = 0.0
                param_server["initial_state"]["state"]["kwargs"]["width"] = 0.5
                param_server["goal_region"]["shape"]["rectangle"]["facecolor"] = colors[j]
                param_server["goal_region"]["shape"]["rectangle"]["opacity"] = 0.6
                planning_problem_set.planning_problem_dict[j].draw(rnd, draw_params=param_server)

            # planning_problem_set.draw(rnd)
            rnd.render()
            plt.savefig(filename + str(i))
            return 0


if __name__ == "__main__":
    assignments_4way = [(0, 3), (2, 0)]
    planning_problem_4way_intersection(assignments=assignments_4way)

    #assignments_highway = [(1, 2, 1, 0), (0, 0, 1, 2)]
    #planning_problem_highway_merging(assignments=assignments_highway)

    assignments_roundabout = [(0, 2, "out"), (90, 0, "in"), (210, 1, "in"), (1, 3, "out")]
    planning_problem_roundabout(assignments=assignments_roundabout)

    # open and visualize

    # 4way intersection
    scenario_name = "ZAM_4_way_intersection-1_1_T-1_problem_set.xml"
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name=scenario_name)
    plot_start_goal(filename="4_way_intersection")

    # highway merging
    # scenario_name = "ZAM_2_lane_merging-1_1_T-1_problem_set.xml"
    # scenario, planning_problem_set = load_commonroad_scenario(scenario_name=scenario_name)
    # plot_start_goal(filename="2_lane_highway_merging")

    # highway merging
    scenario_name = "ZAM_roundabout-1_1_T-1_problem_set.xml"
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name=scenario_name)
    plot_start_goal(filename="roundabout")
