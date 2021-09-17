import math
import numpy as np
from sim.scenarios import load_commonroad_scenario
from sim.simulator import SimContext, Simulator, SimParameters, SimLog
from sim.models.vehicle import VehicleModel, VehicleState
from dg_commons.analysis.metrics_def import MetricEvaluationContext
import matplotlib.pyplot as plt
import json
from dataclasses import fields
from typing import Optional
from sim.models.vehicle_dynamic import VehicleModelDyn, VehicleStateDyn
from commonroad.scenario.obstacle import DynamicObstacle
from games import PlayerName
from sim.scenarios.agent_from_commonroad import infer_lane_from_dyn_obs
import os
from crash.reports import generate_report
from dg_commons.sequence import DgSampledSequence
from sim import SimTime

DT: SimTime = SimTime("0.05")
DT_COMMANDS: SimTime = SimTime("0.1")
assert DT_COMMANDS % DT == SimTime(0)


class TestController:

    def __init__(self, scenario_name: str, vehicle_model: str, metrics, agent, controller,
                 speed_behavior, steering_controller, longitudinal_controller=None, state_estimator=None):

        scenario, _ = load_commonroad_scenario(scenario_name)
        self.lanelet_net = scenario.lanelet_network
        self.agent = agent

        self.controller = controller
        self.longitudinal_controller = longitudinal_controller
        self.speed_behavior = speed_behavior
        self.steering_controller = steering_controller
        self.vehicle_model = vehicle_model
        self.state_estimator = state_estimator

        players, models = {}, {}
        for i, dyn_obs in enumerate(scenario.dynamic_obstacles):
            agent = self._agent_from_dynamic_obstacle(dyn_obs)
            model, estimator = self._model_se_from_dynamic_obstacle(dyn_obs, False)
            agent.set_state_estimator(estimator)
            player_name = PlayerName(f"P{i}")
            players.update({player_name: agent})
            models.update({player_name: model})

        sim_parameters: SimParameters = SimParameters(dt=DT, dt_commands=DT_COMMANDS)
        self.sim_context: SimContext = SimContext(scenario=scenario, models=models, players=players,
                                                  param=sim_parameters, log=SimLog())
        self.metrics = metrics
        self.metrics_context: Optional[MetricEvaluationContext] = None
        self.result = []
        self.simulator: Simulator = Simulator()

    def _agent_from_dynamic_obstacle(self, dyn_obs: DynamicObstacle):

        controller = self.controller["Controller"](self.controller["Parameters"])
        speed_behavior = self.speed_behavior["Behavior"]()
        speed_behavior.params = self.speed_behavior["Parameters"]
        steering_controller = self.steering_controller["Controller"]()
        steering_controller.params = self.steering_controller["Parameters"]
        dg_lane = infer_lane_from_dyn_obs(dyn_obs, self.lanelet_net)
        longitudinal_controller = None

        if self.longitudinal_controller:
            longitudinal_controller = self.longitudinal_controller["Controller"]()
            longitudinal_controller.params = self.longitudinal_controller["Parameters"]

        agent = self.agent(dg_lane, controller=controller, speed_behavior=speed_behavior,
                           speed_controller=longitudinal_controller, steer_controller=steering_controller,
                           return_extra=True)

        return agent

    def _model_se_from_dynamic_obstacle(self, dyn_obs: DynamicObstacle, is_dynamic: bool):
        orient_0, orient_1 = dyn_obs.prediction.trajectory.state_list[0].orientation, \
                             dyn_obs.prediction.trajectory.state_list[1].orientation
        vel_0 = dyn_obs.prediction.trajectory.state_list[0].velocity
        dtheta = orient_1 - orient_0
        l = dyn_obs.obstacle_shape.length
        delta_0 = math.atan(l * dtheta / vel_0)

        state_estimator = None
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
            if self.state_estimator:
                state_estimator = self.state_estimator['Estimator'](DT, params=self.state_estimator["Parameters"])

        return model, state_estimator

    def run(self):
        self.simulator.run(self.sim_context)
        name = "Test"

        nominal_velocity = self.speed_behavior["Parameters"].nominal_speed
        dg_lanelets = {}
        states = {}
        commands = {}
        velocities = {}
        for key in self.sim_context.log.keys():
            dg_lanelets[key] = self.sim_context.players[key].ref_lane
            states[key] = self.sim_context.log[key].states
            commands[key] = self.sim_context.log[key].actions
            timestamps = states[key].timestamps
            velocities[key] = DgSampledSequence(timestamps, len(timestamps)*[nominal_velocity])

        self.metrics_context = MetricEvaluationContext(dg_lanelets, states, commands, velocities)

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
                plt.title(metric.description)
                plt.legend()
                plt.savefig(f"fig{i}")
                plt.figure()
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
                value = value.tolist() if type(value) == np.ndarray else value
                res[field.name] = value
                key_str += str(value)
            return res, key_str

        lateral_dict, key_str = dict_key_from_dataclass(self.controller)
        key_string += key_str
        if self.longitudinal_controller:
            longitudinal_dict, key_str = dict_key_from_dataclass(self.longitudinal_controller)
            key_string += key_str
        else:
            longitudinal_dict = {}
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
