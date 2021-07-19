from sim.models.car import VehicleState, VehicleCommands
import numpy as np


def test_vehicle_state_01():
    npstate = np.array([1, 2, 3, 4, 5])
    vstate = VehicleState.from_array(npstate)
    print(vstate)
    np.testing.assert_array_equal(npstate, vstate.as_ndarray())


def test_vehicle_commnads_01():
    npcmds = np.array([1, 2])
    vcommands = VehicleCommands.from_array(npcmds)
    print(vcommands)
    np.testing.assert_array_equal(npcmds, vcommands.as_ndarray())