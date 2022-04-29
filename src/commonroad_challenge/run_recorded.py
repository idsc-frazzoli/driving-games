import matplotlib.pyplot as plt
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblemSet, PlanningProblem
from commonroad.scenario.scenario import Scenario
from commonroad.visualization.mp_renderer import MPRenderer
import matplotlib.animation as animation
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_route_planner.utility.visualization import visualize_route
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.trajectory import Trajectory, State
from commonroad.prediction.prediction import TrajectoryPrediction
from vehiclemodels import parameters_vehicle3
import numpy as np



def plot_scenario_dynamic(scenario: Scenario,
                          planning_problem_set: PlanningProblemSet,
                          ego_vehicle: DynamicObstacle,
                          show_plot: bool = False):
    fig = plt.figure(figsize=(25, 10))
    rnd = MPRenderer()

    def init_plot():
        scenario.draw(rnd, draw_params={'time_begin': 0})
        planning_problem_set.draw(rnd)
        return rnd.render()

    def update_plot(i):
        scenario.draw(rnd, draw_params={'time_begin': i})
        planning_problem_set.draw(rnd)
        ego_vehicle.draw(rnd, draw_params={'time_begin': i, 'dynamic_obstacle': {
            'vehicle_shape': {'occupancy': {'shape': {'rectangle': {
                'facecolor': 'g'}}}}}})
        return rnd.render()

    anim = animation.FuncAnimation(fig, update_plot)
    if show_plot:
        plt.show()


def generate_route(scenario: Scenario, planning_problem: PlanningProblem, plot_route: bool = False):
    route_planner = RoutePlanner(scenario, planning_problem, backend=RoutePlanner.Backend.NETWORKX_REVERSED)
    # plan routes, save multiple routes as list in candidate holder
    candidate_holder = route_planner.plan_routes()

    # here we retrieve the first route
    # option 1: retrieve first route
    route = candidate_holder.retrieve_first_route()
    # option 2: retrieve all routes
    # list_routes, num_route_candidates = candidate_holder.retrieve_all_routes()
    # option 3: retrieve the best route by orientation metric
    # route = candidate_holder.retrieve_best_route_by_orientation()

    # retrieve reference path from route
    ref_path = route.reference_path
    if plot_route:
        visualize_route(route, draw_route_lanelets=True, draw_reference_path=True, size_x=6)


def get_ego_trajectory(planning_problem: PlanningProblem, N: int, v_ego: float, ds: float) -> Trajectory:
    initial_state = planning_problem.initial_state
    initial_pos = initial_state.position
    state_list = [initial_state]
    for i in range(1, N):
        orientation = initial_state.orientation
        # compute new position
        # add new state to state_list
        state_list.append(State(**{'position': np.array([initial_pos[0] + ds * i * np.cos(orientation),
                                                         initial_pos[1] + ds * i * np.sin(orientation)]),
                                   'orientation': orientation,
                                   'time_step': i,
                                   # 'velocity': v_ego * np.cos(orientation),
                                   # 'velocity_y': v_ego * np.sin(orientation)
                                   'velocity': (ds/0.1) * np.cos(orientation),
                                   'velocity_y': (ds/0.1) * np.sin(orientation)}))

    # create the planned trajectory starting at time step 1
    ego_vehicle_trajectory = Trajectory(initial_time_step=1, state_list=state_list[1:])
    # create the prediction using the planned trajectory and the shape of the ego vehicle
    return ego_vehicle_trajectory


def return_dyn_obstacle_ego(initial_state: State, ego_trajectory: Trajectory):
    vehicle3 = parameters_vehicle3.parameters_vehicle3()
    ego_vehicle_shape = Rectangle(length=vehicle3.l, width=vehicle3.w)
    ego_vehicle_prediction = TrajectoryPrediction(trajectory=ego_trajectory,
                                                  shape=ego_vehicle_shape)

    # the ego vehicle can be visualized by converting it into a DynamicObstacle
    ego_vehicle_type = ObstacleType.CAR
    ego_vehicle = DynamicObstacle(obstacle_id=100, obstacle_type=ego_vehicle_type,
                                  obstacle_shape=ego_vehicle_shape, initial_state=initial_state,
                                  prediction=ego_vehicle_prediction)

    return ego_vehicle
    # plot the scenario and the ego vehicle for each time step


def check_drivability(ego_trajectory: Trajectory, ego_vehicle: DynamicObstacle):
    def collision_checker(ego_vehicle: DynamicObstacle):
        from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_checker, \
            create_collision_object

        # create collision checker from scenario
        cc = create_collision_checker(scenario)

        # create ego vehicle collision object
        ego_vehicle_co = create_collision_object(ego_vehicle)

        # check if ego vehicle collides
        res = cc.collide(ego_vehicle_co)
        print('Collision between the ego vehicle and other static/dynamic obstacles: %s' % res)

        def road_compliance():
            from commonroad_dc.boundary.boundary import create_road_boundary_obstacle

            # create the road boundary
            _, road_boundary = create_road_boundary_obstacle(scenario)

            # add road boundary to collision checker
            cc.add_collision_object(road_boundary)

            # Again: check if ego vehicle collides
            res = cc.collide(ego_vehicle_co)
            print('Collision between the ego vehicle and the road boundary: %s' % res)
            return res
        return res, road_compliance()

    def kinematic_feasibility():
        import commonroad_dc.feasibility.feasibility_checker as feasibility_checker
        from commonroad_dc.feasibility.vehicle_dynamics import VehicleDynamics, VehicleType

        # set time step as scenario time step
        dt = scenario.dt

        # choose vehicle model (here kinematic single-track model)
        vehicle_dynamics = VehicleDynamics.KS(VehicleType.FORD_ESCORT)

        # check feasibility of planned trajectory for the given vehicle model
        feasible, reconstructed_inputs = feasibility_checker.trajectory_feasibility(trajectory=ego_trajectory,
                                                                                    vehicle_dynamics=vehicle_dynamics,
                                                                                    dt=dt)
        print('The planned trajectory is feasible: %s' % feasible)
        return feasible

    collision, road_compliance = collision_checker(ego_vehicle)
    feasible_traj = kinematic_feasibility()


if __name__ == "__main__":
    # generate path of the file to be opened
    # file_path = "/media/leon/Extreme SSD1/MT/scenarios_phase_1/DEU_Flensburg-1_1_T-1.xml"
    file_path = "/media/leon/Extreme SSD1/MT/scenarios_phase_1/USA_Lanker-1_15_T-1.xml"
    # file_path = "/media/leon/Extreme SSD1/MT/scenarios_phase_1/DEU_Lohmar-13_1_T-1.xml"
    # read in the scenario and planning problem set
    show_plots = True
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    generate_route(scenario=scenario, planning_problem=planning_problem, plot_route=show_plots)
    ego_traj = get_ego_trajectory(planning_problem, N=40, v_ego=10, ds=1.0)
    ego_dyn_obstacle = return_dyn_obstacle_ego(initial_state=planning_problem.initial_state, ego_trajectory=ego_traj)
    check_drivability(ego_trajectory=ego_traj, ego_vehicle=ego_dyn_obstacle)
    plot_scenario_dynamic(scenario, planning_problem_set, show_plot=show_plots, ego_vehicle=ego_dyn_obstacle)
