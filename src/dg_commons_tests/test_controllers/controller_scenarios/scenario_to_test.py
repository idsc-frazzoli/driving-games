from dataclasses import dataclass
from typing import Optional, List
from sim.scenarios.utils import load_commonroad_scenario
from commonroad.scenario.scenario import Scenario
from dg_commons_tests.test_controllers.controller_scenarios.utils import race_track_generate_dyn_obs


@dataclass
class ScenarioData:
    scenario_name: str

    fig_name: str

    cars_idx: Optional[List[int]] = None

    simulation_time: float = 6

    sp: float = 0

    def on_init(self):
        self.scenario: Scenario
        self.scenario, _ = load_commonroad_scenario(self.scenario_name)

        if self.scenario_name == "DEU_Hhr-1_1":
            dyn_obs = race_track_generate_dyn_obs(self.scenario, starting_position=self.sp)
            self.scenario.add_objects(dyn_obs[0])

        if self.cars_idx is not None:
            n_cars = len(self.scenario.dynamic_obstacles)
            assert all([idx <= n_cars for idx in self.cars_idx])


scenarios = {
    "lane_change_left": ScenarioData("ZAM_Zip-1_60_T-1", "lane_change_left", [1], simulation_time=7),
    "turn_90_right": ScenarioData("ARG_Carcarana-1_1_T-1", "turn_90_right", [1], simulation_time=7),
    "turn_90_left": ScenarioData("DEU_Muehlhausen-2_2_T-1", "turn_90_left", [3], simulation_time=5),
    "small_snake": ScenarioData("ZAM_Tjunction-1_320_T-1", "small_snake", [3], simulation_time=6),
    "u-turn": ScenarioData("DEU_Hhr-1_1", "u-turn", [0],  simulation_time=30, sp=43),
    "left_cont_curve": ScenarioData("DEU_Hhr-1_1", "lef_cont_curve", [0], simulation_time=15, sp=60),
    "race": ScenarioData("DEU_Hhr-1_1", "race", [0],  simulation_time=180)
}
