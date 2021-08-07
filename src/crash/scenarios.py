from decimal import Decimal as D
from typing import Mapping

from numpy import deg2rad

from games import PlayerName
from sim.agent import NPAgent
from sim.models.vehicle import VehicleState, VehicleModel, VehicleCommands
from sim.models.vehicle_dynamic import VehicleStateDyn, VehicleModelDyn
from sim.simulator import SimContext
from sim.simulator_structures import SimTime, SimParameters, SimulationLog

__all__ = ["get_scenario_01", "get_scenario_02"]


def get_scenario_01() -> SimContext:
    P1, P2, P3, P4, P5, P6 = PlayerName("P1"), PlayerName("P2"), PlayerName("P3"), PlayerName("P4"), PlayerName(
        "P5"), PlayerName("P6")

    x0_p1 = VehicleStateDyn(x=2, y=18, theta=0, vx=5, delta=0)
    x0_p2 = VehicleStateDyn(x=22, y=6, theta=deg2rad(90), vx=6, delta=0)
    x0_p3 = VehicleStateDyn(x=45, y=22, theta=deg2rad(180), vx=4, delta=0)
    x0_p4 = VehicleStateDyn(x=2, y=15, theta=-0.1, vx=5, delta=0)
    x0_p5 = VehicleStateDyn(x=27, y=4, theta=deg2rad(95), vx=6, delta=0)
    x0_p6 = VehicleStateDyn(x=17, y=35, theta=deg2rad(-90), vx=4, delta=0)
    models = {P1: VehicleModelDyn.default_car(x0_p1),
              P2: VehicleModelDyn.default_bicycle(x0_p2),
              P3: VehicleModelDyn.default_car(x0_p3),
              P4: VehicleModelDyn.default_car(x0_p4),
              P5: VehicleModelDyn.default_bicycle(x0_p5),
              P6: VehicleModelDyn.default_car(x0_p6)}

    commands_input: Mapping[SimTime, VehicleCommands] = {D(0): VehicleCommands(acc=0, ddelta=0),
                                                         D(1): VehicleCommands(acc=1, ddelta=0.1),
                                                         D(2): VehicleCommands(acc=2, ddelta=-0.1),
                                                         D(99): VehicleCommands(acc=0, ddelta=0)}
    commands_input_2: Mapping[SimTime, VehicleCommands] = {D(0): VehicleCommands(acc=0, ddelta=0),
                                                           D(1): VehicleCommands(acc=1, ddelta=0.1),
                                                           D(2): VehicleCommands(acc=-1, ddelta=-0.1),
                                                           D(99): VehicleCommands(acc=0, ddelta=0.2)}
    players = {P1: NPAgent(commands_input),
               P2: NPAgent(commands_input),
               P3: NPAgent(commands_input_2),
               P4: NPAgent(commands_input),
               P5: NPAgent(commands_input),
               P6: NPAgent(commands_input_2)
               }

    return SimContext(map_name="4way-double-intersection-only",
                      models=models,
                      players=players,
                      log=SimulationLog(),
                      param=SimParameters(dt=D(0.01), sim_time_after_collision=D(3), max_sim_time=D(0.5)),
                      )


def get_scenario_02() -> SimContext:
    P1, P2, P3 = PlayerName("P1"), PlayerName("P2"), PlayerName("P3")

    x0_p1 = VehicleState(x=2, y=16, theta=0, vx=5, delta=0)
    x0_p2 = VehicleState(x=22, y=6, theta=deg2rad(90), vx=6, delta=0)
    x0_p3 = VehicleState(x=45, y=22, theta=deg2rad(180), vx=4, delta=0)
    models = {P1: VehicleModel.default_car(x0_p1),
              P2: VehicleModel.default_bicycle(x0_p2),
              P3: VehicleModel.default_car(x0_p3)}

    commands_input: Mapping[SimTime, VehicleCommands] = {D(0): VehicleCommands(acc=0, ddelta=0),
                                                         D(1): VehicleCommands(acc=1, ddelta=0.1),
                                                         D(2): VehicleCommands(acc=2, ddelta=-0.1),
                                                         D(99): VehicleCommands(acc=0, ddelta=0)}
    commands_input_2: Mapping[SimTime, VehicleCommands] = {D(0): VehicleCommands(acc=0, ddelta=0),
                                                           D(1): VehicleCommands(acc=1, ddelta=0.1),
                                                           D(2): VehicleCommands(acc=-1, ddelta=-0.1),
                                                           D(99): VehicleCommands(acc=0, ddelta=0.2)}
    players = {P1: NPAgent(commands_input),
               P2: NPAgent(commands_input),
               P3: NPAgent(commands_input_2)}

    return SimContext(map_name="4way-double-intersection-only",
                      models=models,
                      players=players,
                      log=SimulationLog(),
                      param=SimParameters(dt=D(0.05), sim_time_after_collision=D(2)),
                      )
