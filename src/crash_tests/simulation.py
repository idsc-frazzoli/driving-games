from crash.agent import NPAgent
from decimal import Decimal as D

if __name__ == '__main__':
    agent  = NPAgent({D(0): -1, D(1): 0, D(2): 1, D(3): 2})
    print(agent)

    # initialize all contexts/ agents and simulator


    # todo create an "experiment runner" that initialize stuff and calls the updates steps of the simulator till termination


