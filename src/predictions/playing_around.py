import os
from datetime import datetime
from typing import Mapping, Dict
from numpy import cos
from numpy import sin

from dg_commons import PlayerName, X
from dg_commons.planning.trajectory import Trajectory
from sim import logger
from sim.simulator import SimContext, Simulator
from sim.simulator_structures import *
from sim.models.vehicle import VehicleState
from crash.scenarios import get_scenario_two_lanes
from crash.reports import generate_report


sim_context = get_scenario_two_lanes()
net = sim_context.scenario.lanelet_network
net.remove_lanelet(24)
lanelet_1 = net.find_lanelet_by_id(25)
a=10
print("test")

