from sim.scenarios.utils import load_commonroad_scenario
from sim.scenarios.agent_from_commonroad import *
from dg_commons import PlayerName
from sim.simulator import SimContext, Simulator, SimParameters, SimLog
import os
from crash.reports import generate_report


def visualize_scenario(scenario_name: str):
    scenario, _ = load_commonroad_scenario(scenario_name)
    lanelet_net = scenario.lanelet_network

    players, models = {}, {}
    for i, dyn_obs in enumerate(scenario.dynamic_obstacles):
        if dyn_obs.obstacle_type is not ObstacleType.CAR:
            continue

        model, agent = model_agent_from_dynamic_obstacle(dyn_obs, lanelet_net)
        player_name = PlayerName(f"P{i}")
        players.update({player_name: agent})
        models.update({player_name: model})

    sim_parameters: SimParameters = SimParameters()
    sim_context: SimContext = SimContext(scenario=scenario, models=models, players=players,
                                         param=sim_parameters, log=SimLog())

    simulator: Simulator = Simulator()

    simulator.run(sim_context)
    name = "test_" + scenario_name

    report = generate_report(sim_context)
    # save report
    output_dir = "out"
    report_file = os.path.join(output_dir, f"{name}.html")
    report.to_html(report_file)


scenario_name = "ZAM_Tjunction-1_320_T-1"
visualize_scenario(scenario_name)
