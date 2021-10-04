from sim_tests.controllers_tests.controller_scenarios.scenario_to_test import ScenarioData
from dataclasses import dataclass
from typing import Union
from dg_commons.analysis.metrics import Metrics
from dg_commons.controllers.full_controller_base import VehicleController
from sim_tests.controllers_tests.test_controller import TestController
from typing import List


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
