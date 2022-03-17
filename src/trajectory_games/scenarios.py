import os
from decimal import Decimal as D
from typing import List

import numpy as np

from crash.agents import B2Agent
from dg_commons import PlayerName
from dg_commons.controllers.speed import SpeedController, SpeedControllerParam
from dg_commons.controllers.steer import SteerController, SteerControllerParam
from dg_commons.sim.models import kmh2ms
from dg_commons.sim.models.vehicle_dynamic import VehicleModelDyn, VehicleStateDyn
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.scenarios import load_commonroad_scenario
from dg_commons.sim.scenarios.agent_from_commonroad import dglane_from_position
from dg_commons.sim.scenarios.structures import DgScenario
from dg_commons.sim.simulator import SimContext
from dg_commons.sim.simulator_structures import SimParameters
__all__ = [
    "get_scenario_4_way_crossing_stochastic",
]

from dg_commons_dev.utils import get_project_root_dir

P1, EGO = (
    PlayerName("P1"),
    PlayerName("Ego"),
)

SCENARIOS_DIR = os.path.join(get_project_root_dir(), "scenarios")




def get_scenario_4_way_crossing_stochastic() -> SimContext:
    scenario_name = "ZAM_Zip-1_66_T-1"
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name, SCENARIOS_DIR)

    x0_truck = VehicleStateDyn(x=-98, y=5.35, theta=0.00, vx=kmh2ms(30), delta=0)
    x0_p2 = VehicleStateDyn(x=-105, y=9, theta=0.00, vx=kmh2ms(60), delta=0)
    x0_ego = VehicleStateDyn(x=-115, y=5.3, theta=0.00, vx=kmh2ms(90), delta=0)

    truck_model = VehicleModelDyn.default_truck(x0_truck)
    vg_ego = VehicleGeometry.default_car(color="firebrick")
    ego_model = VehicleModelDyn.default_car(x0_ego)
    ego_model.vg = vg_ego

    models = {P1: truck_model, P2: VehicleModelDyn.default_car(x0_p2), EGO: ego_model}

    net = scenario.lanelet_network
    agents: List[B2Agent] = []
    for agent in models:
        if not models[agent].model_type == "pedestrian":
            x0 = models[agent].get_state()
            p = np.array([x0.x, x0.y])
            dglane = dglane_from_position(p, net)
            sp_controller_param: SpeedControllerParam = SpeedControllerParam(
                setpoint_minmax=models[agent].vp.vx_limits,
                output_minmax=models[agent].vp.acc_limits,
            )
            st_controller_param: SteerControllerParam = SteerControllerParam(
                setpoint_minmax=(-models[agent].vp.delta_max, models[agent].vp.delta_max),
                output_minmax=(-models[agent].vp.acc_max, models[agent].vp.acc_max),
            )
            sp_controller = SpeedController(sp_controller_param)
            st_controller = SteerController(st_controller_param)
            agents.append(B2Agent(dglane, speed_controller=sp_controller, steer_controller=st_controller))
    players = {
        P1: agents[0],
        P2: agents[1],
        EGO: agents[2],
    }

    return SimContext(
        dg_scenario=DgScenario(scenario),
        models=models,
        players=players,
        param=SimParameters(dt=D("0.01"), dt_commands=D("0.1"), sim_time_after_collision=D(4), max_sim_time=D(6)),
    )

