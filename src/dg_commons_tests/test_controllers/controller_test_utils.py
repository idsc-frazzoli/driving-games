from dataclasses import dataclass
from typing import Union
from dg_commons.analysis.metrics import Metrics
from typing import List
import math
import numpy as np
from sim.simulator import SimContext, Simulator, SimParameters, SimLog
from sim.models.vehicle import VehicleModel, VehicleState
from dg_commons.analysis.metrics_def import MetricEvaluationContext
import json
from dataclasses import fields
from typing import Optional
from sim.models.vehicle_dynamic import VehicleModelDyn, VehicleStateDyn
from commonroad.scenario.obstacle import DynamicObstacle
from dg_commons import PlayerName
from sim.scenarios.agent_from_commonroad import infer_lane_from_dyn_obs
import os
from dg_commons.seq.sequence import DgSampledSequence
from sim import SimTime
from dg_commons_tests.test_controllers.controller_scenarios.scenario_to_test import ScenarioData
from dg_commons.controllers.full_controller_base import VehicleController
from dg_commons.controllers.speed import SpeedBehavior
from crash.reports import generate_report


@dataclass
class Select:
    item: Union[VehicleController, type(Metrics), ScenarioData]
    test: bool

    def __post_init__(self):
        if self.test and hasattr(self.item, 'on_init'):
            self.item.on_init()


@dataclass
class TestInstance:
    controller: VehicleController

    metric: List[type(Metrics)]

    scenario: ScenarioData

    def run(self):
        test = TestController(scenario=self.scenario, metrics=self.metric, controller=self.controller)
        test.run()
        test.evaluate_metrics()


DT: SimTime = SimTime("0.05")
DT_COMMANDS: SimTime = SimTime("0.1")
assert DT_COMMANDS % DT == SimTime(0)


class TestController:

    def __init__(self, scenario: ScenarioData, metrics, controller: VehicleController,
                 vehicle_model: Optional[str] = None):

        self.scenario: ScenarioData = scenario
        self.lanelet_net = scenario.scenario.lanelet_network
        self.agent = controller.lf_agent
        self.controller = controller
        self.vehicle_model = vehicle_model

        players, models = {}, {}
        dyn_obstacles = scenario.scenario.dynamic_obstacles
        for i, dyn_obs in enumerate(dyn_obstacles):
            if scenario.cars_idx is None or i in scenario.cars_idx:
                agent = self._agent_from_dynamic_obstacle(dyn_obs)
                model, estimator = self._model_se_from_dynamic_obstacle(dyn_obs, False)
                agent.set_state_estimator(estimator)
                player_name = PlayerName(f"P{i}")
                players.update({player_name: agent})
                models.update({player_name: model})

        sim_parameters: SimParameters = SimParameters(dt=DT, dt_commands=DT_COMMANDS,
                                                      max_sim_time=SimTime(scenario.simulation_time))
        self.sim_context: SimContext = SimContext(scenario=scenario.scenario, models=models, players=players,
                                                  param=sim_parameters, log=SimLog())
        self.metrics = metrics
        self.metrics_context: Optional[MetricEvaluationContext] = None
        self.result = []
        self.simulator: Simulator = Simulator()

        controller_name = self.controller.controller.__name__
        self.output_dir = os.path.join("out", controller_name, self.scenario.fig_name)

    def _agent_from_dynamic_obstacle(self, dyn_obs: DynamicObstacle):
        controller = self.controller.controller(self.controller.controller_params)
        speed_behavior = SpeedBehavior()
        speed_behavior.params = self.controller.speed_behavior_param
        steering_controller = self.controller.steering_controller()
        steering_controller.params = self.controller.steering_controller_params

        dg_lane = infer_lane_from_dyn_obs(dyn_obs, self.lanelet_net)
        longitudinal_controller = None

        if self.controller.longitudinal_controller:
            longitudinal_controller = self.controller.longitudinal_controller()
            longitudinal_controller.params = self.controller.longitudinal_controller_params

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
        delta_0 = math.atan(l * dtheta / vel_0) if vel_0 > 10e-6 else 0

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
            if self.controller.state_estimator:
                state_estimator = self.controller.state_estimator(DT, params=self.controller.state_estimator_params)

        return model, state_estimator

    def run(self):
        self.simulator.run(self.sim_context)
        name = "simulation"

        nominal_velocity = self.controller.speed_behavior_param.nominal_speed
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

        # report = generate_report(self.sim_context)
        # save report
        # report_file = os.path.join(self.output_dir, f"{name}.html")
        # report.to_html(report_file)

    def evaluate_metrics(self):

        if self.metrics is not None:
            self.result = []
            for metric in self.metrics:
                met = metric()
                self.result.append(met.evaluate(self.metrics_context, plot=True, output_dir=self.output_dir))
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

