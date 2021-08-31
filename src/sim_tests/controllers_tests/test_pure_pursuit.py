from sim_tests.controllers_tests.test_controller import TestController
import math
import matplotlib.pyplot as plt
from commonroad.scenario.lanelet import Lanelet
from dg_commons.planning.lanes import DgLanelet, LaneCtrPoint
from dg_commons.controllers.speed import SpeedBehavior, SpeedController, SpeedControllerParam, SpeedBehaviorParam
from dg_commons.controllers.pure_pursuit import PurePursuit, PurePursuitParam
from sim.scenarios import load_commonroad_scenario
from sim.agents.lane_follower import LFAgent
from sim.simulator import SimContext, Simulator, SimParameters, SimulationLog
from sim.models.vehicle import VehicleModel, VehicleState
import numpy as np
from crash.reports import generate_report
import os
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from games import PlayerName
from typing import Optional
from geometry import translation_angle_from_SE2
from sim_tests.controllers_tests.lanelet_generator import LaneletGenerator


def test_pure_pursuit():
    scenario_name: str = "USA_Peach-1_1_T-1"
    # scenario_name: str = "ZAM_Tjunction-1_129_T-1"
    # scenario_name: str = "ARG_Carcarana-1_1_T-1"
    """Name of the chosen scenario"""
    vehicle_speed: float = 5
    """Nominal speed of the vehicle"""
    k_lookahead: float = 1.8
    """Scaling constant for speed dependent params"""
    ddelta_kp: float = 10
    """Proportional gain ddelta with respect to delta error"""
    speed_kp: float = 0.5
    """Propotioanl gain longitudinal speed controller"""
    speed_ki: float = 0.01
    """Integral gain longitudinal speed controller"""

    sp_controller_param: SpeedControllerParam = SpeedControllerParam(kI=speed_ki, kP=speed_kp)
    sp_controller = {"Controller": SpeedController, "Parameters": sp_controller_param}
    """Speed Controller"""
    sp_behavior_param: SpeedBehaviorParam = SpeedBehaviorParam(nominal_speed=vehicle_speed)
    sp_behavior = {"Behavior": SpeedBehavior, "Parameters": sp_behavior_param}
    """Speed behavior"""
    pp_param: PurePursuitParam = PurePursuitParam(k_lookahead=k_lookahead)
    pp_controller = {"Controller": PurePursuit, "Parameters": pp_param}
    """Pure Pursuit Controller"""

    test_pp = TestController(scenario_name, "-", pp_controller, sp_controller, sp_behavior)
    test_pp.run()


test_pure_pursuit()
