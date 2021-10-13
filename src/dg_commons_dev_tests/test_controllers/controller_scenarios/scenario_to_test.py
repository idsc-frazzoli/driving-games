from dataclasses import dataclass
from typing import Optional, List, Dict
from sim.scenarios.utils import load_commonroad_scenario
from commonroad.scenario.scenario import Scenario
from dg_commons_dev_tests.test_controllers.controller_scenarios.utils import race_track_generate_dyn_obs


@dataclass
class ScenarioData:
    scenario_name: str

    fig_name: str

    cars_idx: Optional[List[int]] = None

    simulation_time: float = 6

    race_params: Optional[List[float]] = None

    def on_init(self):
        self.scenario: Scenario
        self.scenario, _ = load_commonroad_scenario(self.scenario_name)

        if self.scenario_name == "DEU_Hhr-1_1":
            dyn_obs = race_track_generate_dyn_obs(self.scenario,
                                                  starting_position=self.race_params[0],
                                                  length_perc=self.race_params[1]) if self.race_params else \
                                                  race_track_generate_dyn_obs(self.scenario)
            self.scenario.add_objects(dyn_obs[0])

        if self.cars_idx is not None:
            n_cars = len(self.scenario.dynamic_obstacles)
            assert all([idx <= n_cars for idx in self.cars_idx])


scenarios: Dict[str, ScenarioData] = {
    "lane_change_left": ScenarioData("ZAM_Zip-1_60_T-1", "lane_change_left", [1], simulation_time=7),
    "turn_90_right": ScenarioData("ARG_Carcarana-1_1_T-1", "turn_90_right", [1], simulation_time=7),
    "turn_90_left": ScenarioData("DEU_Muehlhausen-2_2_T-1", "turn_90_left", [3], simulation_time=5),
    "small_snake": ScenarioData("ZAM_Tjunction-1_320_T-1", "small_snake", [3], simulation_time=6),
    "u-turn": ScenarioData("DEU_Hhr-1_1", "u-turn", [0],  simulation_time=30, race_params=[43, 10]),
    "left_cont_curve": ScenarioData("DEU_Hhr-1_1", "left_cont_curve", [0], simulation_time=15, race_params=[60, 10]),
    "vertical": ScenarioData("USA_Peach-3_2_T-1", "vertical", [4, 7], simulation_time=7),
    "race": ScenarioData("DEU_Hhr-1_1", "race", [0],  simulation_time=180, race_params=[0, 95])
}
