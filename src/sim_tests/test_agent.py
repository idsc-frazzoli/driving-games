from sim.agent import NPAgent
from decimal import Decimal as D
import numpy as np
from sim.simulator_structures import SimObservations


def test_npagent():
    agent = NPAgent({D(0): -1, D(1): 0, D(2): 1, D(3): 2})
    ts_list = [D(i) for i in np.linspace(1, 6, 20)]
    sim_obs = SimObservations({}, D(0))
    for ts in ts_list:
        sim_obs.time = ts
        cmds = agent.get_commands(sim_obs=sim_obs)
        print(f"At {ts:.2f} agent cmds: {cmds}")
