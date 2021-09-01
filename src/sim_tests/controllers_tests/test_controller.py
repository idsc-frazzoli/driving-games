import math
from sim.scenarios import load_commonroad_scenario
from sim.agents.lane_follower import LFAgent
from sim.simulator import SimContext, Simulator, SimParameters, SimulationLog
from sim.models.vehicle import VehicleModel, VehicleState
from sim.models.vehicle_dynamic import VehicleModelDyn, VehicleStateDyn
from crash.reports import generate_report
import os
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from games import PlayerName
from sim.scenarios.agent_from_commonroad import infer_lane_from_dyn_obs
from decimal import Decimal


class TestController:

    def __init__(self, scenario_name: str, vehicle_model: str, lateral_controller, longitudinal_controller,
                 speed_behavior, steering_controller):
        scenario, _ = load_commonroad_scenario(scenario_name)
        self.lanelet_net = scenario.lanelet_network

        self.lateral_controller = lateral_controller
        self.longitudinal_controller = longitudinal_controller
        self.speed_behavior = speed_behavior
        self.steering_controller = steering_controller
        self.vehicle_model = vehicle_model

        players, models = {}, {}
        for i, dyn_obs in enumerate(scenario.dynamic_obstacles):
            agent = self._agent_model_from_dynamic_obstacle(dyn_obs)
            model = TestController._model_from_dynamic_obstacle(dyn_obs, False)
            player_name = PlayerName(f"P{i}")
            players.update({player_name: agent})
            models.update({player_name: model})

        self.sim_context: SimContext = SimContext(scenario=scenario, models=models, players=players,
                                                  param=SimParameters.default(), log=SimulationLog())
        self.simulator: Simulator = Simulator()

    def _agent_model_from_dynamic_obstacle(self, dyn_obs: DynamicObstacle):

        lateral_controller = self.lateral_controller["Controller"]()
        lateral_controller.params = self.lateral_controller["Parameters"]
        longitudinal_controller = self.longitudinal_controller["Controller"]()
        longitudinal_controller.params = self.longitudinal_controller["Parameters"]
        speed_behavior = self.speed_behavior["Behavior"]()
        speed_behavior.params = self.speed_behavior["Parameters"]
        steering_controller = self.steering_controller["Controller"]()
        steering_controller.param = self.steering_controller["Parameters"]

        dg_lane = infer_lane_from_dyn_obs(dyn_obs, self.lanelet_net)
        agent: LFAgent = LFAgent(dg_lane, speed_behavior=speed_behavior, speed_controller=longitudinal_controller,
                                 lateral_controller=lateral_controller, steering_controller=steering_controller)
        return agent

    @staticmethod
    def _model_from_dynamic_obstacle(dyn_obs: DynamicObstacle, is_dynamic: bool):
        orient_0, orient_1 = dyn_obs.prediction.trajectory.state_list[0].orientation, \
                             dyn_obs.prediction.trajectory.state_list[1].orientation
        vel_0 = dyn_obs.prediction.trajectory.state_list[0].velocity
        dtheta = orient_1 - orient_0
        l = dyn_obs.obstacle_shape.length
        delta_0 = math.atan(l * dtheta / vel_0)
        if is_dynamic:
            x0 = VehicleStateDyn(x=dyn_obs.initial_state.position[0], y=dyn_obs.initial_state.position[1],
                                 theta=dyn_obs.initial_state.orientation, vx=dyn_obs.initial_state.velocity,
                                 delta=delta_0, vy=0, dtheta=dtheta)
            model = VehicleModel.default_car(x0=x0)
        else:
            x0 = VehicleState(x=dyn_obs.initial_state.position[0], y=dyn_obs.initial_state.position[1],
                              theta=dyn_obs.initial_state.orientation, vx=dyn_obs.initial_state.velocity,
                              delta=delta_0)
            model = VehicleModel.default_car(x0=x0)

        return model

    def run(self):
        self.simulator.run(self.sim_context)
        name = "notimplemented"

        report = generate_report(self.sim_context)
        # save report
        output_dir = "out"
        report_file = os.path.join(output_dir, f"{name}.html")
        report.to_html(report_file)