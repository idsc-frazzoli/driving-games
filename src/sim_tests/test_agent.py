from decimal import Decimal as D

import numpy as np

from sim.agent import NPAgent
from sim.simulator_structures import SimObservations


def test_npagent():
    agent = NPAgent({D(0): 0, D(1): 1, D(2): 2, D(3): 3})
    ts_list = [D(i) for i in np.linspace(0, 6, 20)]
    sim_obs = SimObservations({}, D(0))
    for ts in ts_list:
        sim_obs.time = ts
        cmds = agent.get_commands(sim_obs=sim_obs)
        print(f"At {ts:.2f} agent cmds: {cmds}")
