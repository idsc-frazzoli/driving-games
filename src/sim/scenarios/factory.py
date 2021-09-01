from typing import Optional

from games import PlayerName
from sim import logger, SimLog, SimParameters
from sim.scenarios import load_commonroad_scenario
from sim.scenarios.agent_from_commonroad import model_agent_from_dynamic_obstacle, NotSupportedConversion
from sim.simulator import SimContext


def get_scenario_commonroad_replica(scenario_name: str, sim_param: Optional[SimParameters] = None) -> SimContext:
    """
    This functions load a commonroad scenario and tries to convert the dynamic obstacles into the Model/Agent paradigm
    used by the driving-game simulator.
    :param scenario_name:
    :param sim_param:
    :return:
    """
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name)
    players, models = {}, {}
    for i, dyn_obs in enumerate(scenario.dynamic_obstacles):
        try:
            model, agent = model_agent_from_dynamic_obstacle(dyn_obs, scenario.lanelet_network)
            playername = PlayerName(f"P{i}")
            players.update({playername: agent})
            models.update({playername: model})
        except NotSupportedConversion as e:
            logger.warn("Unable to convert commonroad dynamic obstacle due to " + e.args[0] + " skipping...")
    logger.info(f"Managed to load {len(players)}")
    sim_param = SimParameters() if sim_param is None else sim_param
    return SimContext(scenario=scenario,
                      models=models,
                      players=players,
                      log=SimLog(),
                      param=sim_param,
                      )
