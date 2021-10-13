from sim.simulator import SimContext, Simulator, SimParameters, SimLog
from sim.models.vehicle import VehicleModel, VehicleState
from dataclasses import fields, dataclass
from sim.models.vehicle_dynamic import VehicleModelDyn, VehicleStateDyn
from commonroad.scenario.obstacle import DynamicObstacle
from sim.scenarios.agent_from_commonroad import infer_lane_from_dyn_obs
from dg_commons.seq.sequence import DgSampledSequence
from sim import SimTime
from dg_commons_dev_tests.test_controllers.controller_scenarios.scenario_to_test import ScenarioData
from dg_commons_dev_tests.test_controllers.controllers_to_test import *
from dg_commons_dev.analysis.metrics import *
from dg_commons_dev.analysis.metrics_def import *
import matplotlib.pyplot as plt
from typing import List
import os
import math
from dg_commons_dev.controllers.full_controller_base import VehicleController
from dg_commons_dev.controllers.speed import SpeedBehavior
import numpy as np


class Verbosity:
    def __init__(self, val: int):
        assert val in [0, 1, 2]
        self._val = val

    @property
    def val(self):
        return self._val


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
        return test.result


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

        self.output_dir = os.path.join("out", self.controller.folder_name, self.scenario.fig_name)

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
                state_estimator = self.controller.state_estimator(params=self.controller.state_estimator_params)

        return model, state_estimator

    def run(self):
        self.simulator.run(self.sim_context)
        name = "simulation"

        nominal_velocity = self.controller.speed_behavior_param.nominal_speed
        dg_lanelets = {}
        states = {}
        commands = {}
        velocities = {}
        betas = {}
        dt_commands = {}
        for key in self.sim_context.log.keys():
            dg_lanelets[key] = self.sim_context.players[key].ref_lane
            states[key] = self.sim_context.log[key].states
            commands[key] = self.sim_context.log[key].actions
            dt_timestamps = states[key].timestamps
            dt_commands_timestamps = self.sim_context.log[key].actions.timestamps
            velocities[key] = DgSampledSequence(dt_timestamps, len(dt_timestamps)*[nominal_velocity])
            betas[key] = DgSampledSequence(dt_commands_timestamps, self.sim_context.players[key].betas)
            dt_commands[key] = DgSampledSequence(dt_commands_timestamps, self.sim_context.players[key].dt_commands)

        self.metrics_context = MetricEvaluationContext(dg_lanelets, states, commands,
                                                       velocities, dt_commands, betas)

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


class DataCollect:
    def __init__(self, keys):
        self.data = {key: [] for key in keys}
        self.controllers = []
        self.scenarios = []
        self.timing = {}

    def collect_data(self, dict_name, result, scenario_fig_name, percentage, controller_folder_name, verbosity):
        self.timing[dict_name] = {}
        helper = {key: [] for key in self.data.keys()}
        if verbosity.val > 0:
            print("[Testing]...")
            print("[Controller]...", controller_folder_name)
            print("[Scenario]...", scenario_fig_name)
            print("[Percentage]...", percentage, "%")
            for key in result[0].keys():
                print()
                print("[Results for {}]...".format(key))
                for item in result:
                    if item[key].title == DTForCommand.__name__:
                        delta = round(float(np.average(np.array(item[key].incremental.values))), 4)
                        print(DTForCommand.__name__, " = {}".format(delta))
                        helper[DTForCommand.__name__].append(delta)
                    else:
                        helper[item[key].title].append(item[key].total)
                        print(item[key])

        for key in self.data.keys():
            self.data[key].append(np.average(np.array(helper[key])))
        self.controllers.append(controller_folder_name)
        self.scenarios.append(scenario_fig_name)
        self.timing[dict_name][scenario_fig_name] = np.average(np.array(helper[DTForCommand.__name__]))

    def show_data(self, verbosity):
        name = "OVERALL STATISTICS"
        output_dir = os.path.join("out", "simulation_timing_statistics")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        def show_results(ch_name, data, ch_controllers, plots=False):
            if verbosity.val > 1:
                print(ch_name)

            for key in data.keys():
                if verbosity.val > 1:
                    minimum = min(data[key])
                    idx_min = data[key].index(minimum)
                    maximum = max(data[key])
                    idx_max = data[key].index(maximum)
                    print("Total/Average {}: {}".format(key, round(sum(data[key]) / len(data[key]), 3)))
                    max_scene = ", Scenario: " + self.scenarios[idx_max] if ch_name == "OVERALL STATISTICS" else ""
                    print("Max {} is {} for: ".format(key, round(maximum, 3)),
                          " Controller: {}{}".format(ch_controllers[idx_max], max_scene))
                    min_scene = ", Scenario: " + self.scenarios[idx_min] if name == "OVERALL STATISTICS" else ""
                    print("Min {} is {} for: ".format(key, round(minimum, 3)),
                          " Controller: {}{}".format(ch_controllers[idx_min], min_scene))
                    print()
                if plots:
                    plt.hist(data[key])
                    plt.title(ch_name + " " + key)
                    plt.savefig(os.path.join(output_dir, ch_name + key))
                    plt.clf()

        scenario_list = list(dict.fromkeys(self.scenarios))
        show_results(name, self.data, self.controllers)
        for scene in scenario_list:
            helper = [i for i, t in enumerate(self.scenarios) if t == scene]
            temp_data = {key: [self.data[key][i] for i in helper] for key in self.data.keys()}
            temp_controller = [self.controllers[i] for i in helper]
            name = "STATISTICS ABOUT " + scene.upper()
            show_results(name, temp_data, temp_controller)
