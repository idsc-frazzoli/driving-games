from sim.agent import NPAgent
from decimal import Decimal as D

from sim.simulator import Simulator, SimContext

if __name__ == '__main__':

    sim = Simulator()
    print(sim)
    # initialize all contexts/ agents and simulator

    # todo create an "experiment runner" that initialize stuff and calls the updates steps of the simulator till termination
