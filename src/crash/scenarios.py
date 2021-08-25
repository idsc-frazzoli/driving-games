from decimal import Decimal as D
from math import pi
from typing import List

import numpy as np
from commonroad.scenario.lanelet import Lanelet
from numpy import deg2rad

from dg_commons import DgSampledSequence
from dg_commons.planning.lanes import DgLanelet
from games import PlayerName
from sim.agents.agent import NPAgent
from sim.agents.lane_follower import LFAgent
from sim.models.pedestrian import PedestrianState, PedestrianModel, PedestrianCommands
from sim.models.vehicle import VehicleCommands
from sim.models.vehicle_dynamic import VehicleStateDyn, VehicleModelDyn
from sim.scenarios import load_commonroad_scenario
from sim.scenarios.agent_from_commonroad import npAgent_from_dynamic_obstacle
from sim.simulator import SimContext
from sim.simulator_structures import SimParameters, SimulationLog

__all__ = ["get_scenario_01", "get_scenario_az_01"]

P1, P2, P3, P4, P5, P6, P7 = PlayerName("P1"), PlayerName("P2"), PlayerName("P3"), PlayerName("P4"), PlayerName(
    "P5"), PlayerName("P6"), PlayerName("P7")


def get_scenario_01() -> SimContext:
    scenario_name = "USA_Lanker-1_1_T-1.xml"
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name)
    dyn_obs = scenario.dynamic_obstacles[2]
    agent, model = npAgent_from_dynamic_obstacle(dyn_obs, scenario.dt)

    x0_p1 = VehicleStateDyn(x=2, y=18, theta=0, vx=5, delta=0)
    x0_p2 = VehicleStateDyn(x=22, y=6, theta=deg2rad(90), vx=6, delta=0)
    x0_p3 = VehicleStateDyn(x=45, y=22, theta=deg2rad(180), vx=7, delta=0)
    x0_p4 = VehicleStateDyn(x=2, y=15, theta=-0.1, vx=5, delta=0)
    x0_p5 = VehicleStateDyn(x=27, y=4, theta=deg2rad(95), vx=8, delta=0)
    x0_p6 = VehicleStateDyn(x=17, y=35, theta=deg2rad(-90), vx=6, delta=0)
    x0_p7 = PedestrianState(x=11, y=32, theta=deg2rad(-90), vx=4)
    models = {P1: VehicleModelDyn.default_car(x0_p1),
              P2: VehicleModelDyn.default_bicycle(x0_p2),
              P3: model,
              P4: VehicleModelDyn.default_car(x0_p4),
              P5: VehicleModelDyn.default_bicycle(x0_p5),
              P6: VehicleModelDyn.default_car(x0_p6),
              P7: PedestrianModel.default(x0_p7)}

    timestamps = [0, 1, 1.5, 3]
    commands = [(0, 0), (1, 0.2), (-1, -0.1), (2, 0.1), ]
    vehicle_commands = [VehicleCommands(acc=acc, ddelta=ddelta) for acc, ddelta in commands]

    commands_plan = DgSampledSequence[VehicleCommands](timestamps=timestamps, values=vehicle_commands)

    ped_commands_plan = DgSampledSequence[PedestrianCommands](
        timestamps=[0, 4], values=[PedestrianCommands(acc=0, dtheta=0), PedestrianCommands(acc=0, dtheta=-0.1)])

    players = {P1: NPAgent(commands_plan),
               P2: NPAgent(commands_plan),
               P3: agent,
               P4: NPAgent(commands_plan),
               P5: NPAgent(commands_plan),
               P6: NPAgent(commands_plan),
               P7: NPAgent(ped_commands_plan)
               }

    return SimContext(scenario=scenario,
                      models=models,
                      players=players,
                      log=SimulationLog(),
                      param=SimParameters(
                          dt=D(0.02), dt_commands=D(0.1), sim_time_after_collision=D(4), max_sim_time=D(5)),
                      )


def get_scenario_az_01() -> SimContext:
    scenario_name = "USA_Peach-1_1_T-1.xml"
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name)
    scenario.translate_rotate(translation=np.array([0, 0]), angle=-pi / 2)
    x0_p3 = PedestrianState(x=-15, y=-18, theta=deg2rad(90), vx=0)
    x0_p1 = VehicleStateDyn(x=-24, y=-8, theta=0.1, vx=5, delta=0)
    x0_p2 = VehicleStateDyn(x=-25, y=-11, theta=0.1, vx=6, delta=0)
    x0_p4 = VehicleStateDyn(x=-26, y=-14, theta=0.1, vx=5, delta=0)
    x0_p5 = VehicleStateDyn(x=-10, y=-4, theta=deg2rad(188), vx=7, delta=0)

    models = {P1: VehicleModelDyn.default_car(x0_p1),
              P2: VehicleModelDyn.default_car(x0_p2),
              P3: PedestrianModel.default(x0_p3),
              P4: VehicleModelDyn.default_car(x0_p4),
              P5: VehicleModelDyn.default_car(x0_p5),
              }

    ped_commands_plan = DgSampledSequence[PedestrianCommands](
        timestamps=[0, 0.5, 1], values=[PedestrianCommands(acc=0, dtheta=0),
                                        PedestrianCommands(acc=3, dtheta=0.1),
                                        PedestrianCommands(acc=3, dtheta=0)])

    net = scenario.lanelet_network
    agents: List[LFAgent] = []
    for x0 in [x0_p1, x0_p2, x0_p4, x0_p5]:
        p = np.array([x0.x, x0.y])
        lane_id = net.find_lanelet_by_position([p, ])
        assert len(lane_id[0]) > 0, p
        lane = net.find_lanelet_by_id(lane_id[0][0])
        merged_lane = Lanelet.all_lanelets_by_merging_successors_from_lanelet(
            lanelet=lane, network=net)[0][0]
        agents.append(LFAgent(DgLanelet.from_commonroad_lanelet(merged_lane)))
    players = {P1: agents[0],
               P2: agents[1],
               P3: NPAgent(ped_commands_plan),
               P4: agents[2],
               P5: agents[3],
               }

    return SimContext(scenario=scenario,
                      models=models,
                      players=players,
                      log=SimulationLog(),
                      param=SimParameters(
                          dt=D(0.02), dt_commands=D(0.1), sim_time_after_collision=D(4), max_sim_time=D(6)),
                      )
