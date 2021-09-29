from dataclasses import dataclass
from typing import Optional, List
from sim.scenarios.utils import load_commonroad_scenario
from commonroad.scenario.scenario import Scenario


@dataclass
class ScenarioData:
    scenario_name: str

    fig_name: str

    cars_idx: Optional[List[int]] = None

    simulation_time: float = 6

    def __post_init__(self):
        self.scenario: Scenario
        self.scenario, _ = load_commonroad_scenario(self.scenario_name)

        if self.cars_idx is not None:
            n_cars = len(self.scenario.dynamic_obstacles)
            assert all([idx <= n_cars for idx in self.cars_idx])


scenarios = {
    "lane_change_left": ScenarioData("ZAM_Zip-1_60_T-1", "lane_change_left", [1], simulation_time=7),
    "turn_90_right": ScenarioData("ARG_Carcarana-1_1_T-1", "turn_90_right", [1], simulation_time=7),
    "turn_90_left": ScenarioData("DEU_Muehlhausen-2_2_T-1", "turn_90_left", [3], simulation_time=5),
    "small_snake": ScenarioData("ZAM_Tjunction-1_320_T-1", "small_snake", [3], simulation_time=6)
}
