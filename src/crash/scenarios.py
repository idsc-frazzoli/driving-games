from decimal import Decimal as D

from numpy import deg2rad

from dg_commons import DgSampledSequence
from games import PlayerName
from sim.agents.agent import NPAgent
from sim.models.pedestrian import PedestrianState, PedestrianModel, PedestrianCommands
from sim.models.vehicle import VehicleCommands
from sim.models.vehicle_dynamic import VehicleStateDyn, VehicleModelDyn
from sim.scenarios import load_commonroad_scenario
from sim.scenarios.agent_from_commonroad import npAgent_from_dynamic_obstacle
from sim.simulator import SimContext
from sim.simulator_structures import SimParameters, SimulationLog

__all__ = ["get_scenario_01", "get_scenario_02", "get_scenario_03"]


def get_scenario_01() -> SimContext:
    P1, P2, P3, P4, P5, P6, P7 = PlayerName("P1"), PlayerName("P2"), PlayerName("P3"), PlayerName("P4"), PlayerName(
        "P5"), PlayerName("P6"), PlayerName("P7")

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

    return SimContext(scenario_name=scenario_name,
                      models=models,
                      players=players,
                      log=SimulationLog(),
                      param=SimParameters(
                          dt=D(0.02), dt_commands=D(0.1), sim_time_after_collision=D(4), max_sim_time=D(5)),
                      )


def get_scenario_03() -> SimContext:
    P1 = PlayerName("P1")

    x0_p1 = VehicleStateDyn(x=2, y=18, theta=0, vx=5, delta=0)

    models = {P1: VehicleModelDyn.default_car(x0_p1)}

    commands_input: Mapping[SimTime, VehicleCommands] = {D(0): VehicleCommands(acc=0, ddelta=0),
                                                         D(1): VehicleCommands(acc=1, ddelta=0.3),
                                                         D(2): VehicleCommands(acc=2, ddelta=-0.6),
                                                         D(99): VehicleCommands(acc=0, ddelta=0)}

    players = {P1: NPAgent(commands_input)}

    return SimContext(map_name="4way-double-intersection-only",
                      models=models,
                      players=players,
                      log=SimulationLog(),
                      param=SimParameters(dt=D(0.01), sim_time_after_collision=D(3), max_sim_time=D(3)),
                      )