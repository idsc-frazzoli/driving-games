from typing import Mapping

from games import PlayerName
from sim.agent import NPAgent
from decimal import Decimal as D

from sim.models.car import VehicleState, VehicleModel, VehicleCommands
from sim.simulator import Simulator, SimContext
from sim.simulator_structures import SimTime, SimParameters


def get_scenario_01() -> SimContext:
    P1, P2, P3 = PlayerName("P1"), PlayerName("P2"), PlayerName("P3")

    x0_p1 = VehicleState(x=0, y=0, theta=0, vx=10, delta=0)
    x0_p2 = VehicleState(x=4, y=0, theta=0, vx=10, delta=0)
    x0_p3 = VehicleState(x=7, y=0, theta=0, vx=10, delta=0)
    models = {P1: VehicleModel.default_car(x0_p1),
              P2: VehicleModel.default_car(x0_p2),
              P3: VehicleModel.default_car(x0_p3)}

    commands_input: Mapping[SimTime, VehicleCommands] = {D(0): VehicleCommands(acc=0, ddelta=0),
                                                         D(1): VehicleCommands(acc=1, ddelta=0.1),
                                                         D(2): VehicleCommands(acc=2, ddelta=-0.1),
                                                         D(99): VehicleCommands(acc=0, ddelta=0)}
    commands_input_2: Mapping[SimTime, VehicleCommands] = {D(0): VehicleCommands(acc=0, ddelta=0),
                                                         D(1): VehicleCommands(acc=1, ddelta=0.1),
                                                         D(2): VehicleCommands(acc=-1, ddelta=-0.1),
                                                         D(99): VehicleCommands(acc=0, ddelta=0)}
    players = {P1: NPAgent(commands_input),
               P2: NPAgent(commands_input),
               P3: NPAgent(commands_input_2)}

    return SimContext(models=models,
                      players=players,
                      log={},
                      param=SimParameters(), map=None
                      )


if __name__ == '__main__':
    sim = Simulator()
    sim_context = get_scenario_01()
    sim_context = sim.run(sim_context)
    # initialize all contexts/ agents and simulator

    # todo create an "experiment runner" that initialize stuff and calls the updates steps of the simulator till termination
