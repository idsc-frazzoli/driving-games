import math
from sim.scenarios import load_commonroad_scenario
from sim.agents.lane_follower import LFAgent
from sim.simulator import SimContext, Simulator, SimParameters, SimLog
from sim.models.vehicle import VehicleModel, VehicleState
from sim.models.vehicle_dynamic import VehicleModelDyn, VehicleStateDyn
from crash.reports import generate_report
import os
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from games import PlayerName
from sim.scenarios.agent_from_commonroad import infer_lane_from_dyn_obs
from decimal import Decimal
from dg_commons.analysis.metrics_def import MetricEvaluationContext, Metric
from typing import Optional, List
import matplotlib.pyplot as plt
import json
from dataclasses import dataclass, fields


class TestController:

    def __init__(self, scenario_name: str, vehicle_model: str, lateral_controller, longitudinal_controller,
                 speed_behavior, steering_controller, metrics):
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
            model = TestController._model_from_dynamic_obstacle(dyn_obs, True)
            player_name = PlayerName(f"P{i}")
            players.update({player_name: agent})
            models.update({player_name: model})

        self.sim_context: SimContext = SimContext(scenario=scenario, models=models, players=players,
                                                  param=SimParameters(), log=SimLog())
        self.simulator: Simulator = Simulator()

        self.metrics = metrics
        self.metrics_context: Optional[MetricEvaluationContext] = None
        self.result = []

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
            model = VehicleModelDyn.default_car(x0=x0)
        else:
            x0 = VehicleState(x=dyn_obs.initial_state.position[0], y=dyn_obs.initial_state.position[1],
                              theta=dyn_obs.initial_state.orientation, vx=dyn_obs.initial_state.velocity,
                              delta=delta_0)
            model = VehicleModel.default_car(x0=x0)

        return model

    def run(self):
        self.simulator.run(self.sim_context)
        name = "Test"

        dg_lanelets = {}
        states = {}
        commands = {}
        for key in self.sim_context.log.keys():
            dg_lanelets[key] = self.sim_context.players[key].ref_lane
            states[key] = self.sim_context.log[key].states
            commands[key] = self.sim_context.log[key].actions

        self.metrics_context = MetricEvaluationContext(dg_lanelets, states, commands)

        report = generate_report(self.sim_context)
        # save report
        output_dir = "out"
        report_file = os.path.join(output_dir, f"{name}.html")
        report.to_html(report_file)

    def evaluate_metrics_test(self):
        if self.metrics is not None:
            for i, metric in enumerate(self.metrics):
                for player in self.sim_context.players.keys():
                    met = metric()
                    res = met.evaluate(self.metrics_context)
                    plt.plot(res[player].incremental.timestamps, res[player].incremental.values, label=player)
                plt.savefig(f"fig{i}")
        else:
            print("No Metric to Evaluate")

    def evaluate_metrics(self):
        if self.metrics is not None:
            self.result = []
            for metric in self.metrics:
                met = metric()
                self.result.append(met.evaluate(self.metrics_context))
        else:
            print("No Metric to Evaluate")

    def to_json(self):
        key_string = ""

        def dict_key_from_dataclass(data):
            res = {"Name": data["Name"]}
            key_str = str(data["Name"])
            for field in fields(data["Parameters"]):
                value = getattr(data["Parameters"], field.name)
                res[field.name] = value
                key_str += str(value)
            return res, key_str

        lateral_dict, key_str = dict_key_from_dataclass(self.lateral_controller)
        key_string += key_str
        longitudinal_dict, key_str = dict_key_from_dataclass(self.longitudinal_controller)
        key_string += key_str
        steering_dict, key_str = dict_key_from_dataclass(self.steering_controller)
        key_string += key_str

        metric_dict = {}
        for i, metric in enumerate(self.metrics):
            metric_dict["Name"] = metric.description
            key_string += str(metric.description)
            result = self.result[i]
            player_dict = {}
            for player in self.sim_context.players.keys():
                player_dict["Total"] = result[player].total
                player_dict["IncrementalT"] = [float(i) for i in result[player].incremental.timestamps]
                player_dict["IncrementalV"] = result[player].incremental.values
                player_dict["CumulativeT"] = [float(i) for i in result[player].cumulative.timestamps]
                player_dict["CumulativeV"] = result[player].cumulative.values
                metric_dict[f"Results {player}"] = player_dict

        json_dict = {"LateralController": lateral_dict, "LongitudinalController": longitudinal_dict,
                     "SteeringController": steering_dict, "Results": metric_dict}

        json_object = json.dumps(json_dict, indent=4)
        with open("results.json", "w") as outfile:
            outfile.write(json_object)
        key_string = key_string.replace(" ", "")
        print(key_string)
