from decimal import Decimal as D
from math import pi
from typing import List

import numpy as np
from commonroad.scenario.lanelet import Lanelet
from geometry import xytheta_from_SE2
from numpy import deg2rad, linspace

from dg_commons import DgSampledSequence
from dg_commons.controllers.speed import SpeedController, SpeedControllerParam
from dg_commons.controllers.steer import SteerControllerParam, SteerController
from dg_commons.planning.lanes import DgLanelet
from games import PlayerName
from sim import SimTime
from sim.agents.agent import NPAgent
from sim.agents.lane_follower import LFAgent
from sim.models import kmh2ms, CAR, TRUCK
from sim.models.pedestrian import PedestrianState, PedestrianModel, PedestrianCommands
from sim.models.vehicle_dynamic import VehicleStateDyn, VehicleModelDyn
from sim.models.vehicle_structures import VehicleGeometry
from sim.scenarios import load_commonroad_scenario
from sim.scenarios.factory import get_scenario_commonroad_replica
from sim.simulator import SimContext
from sim.simulator_structures import SimParameters

__all__ = ["get_scenario_bicycle", "get_scenario_illegal_turn", "get_scenario_suicidal_pedestrian",
           "get_scenario_racetrack_test", "get_two_lanes_scenario"]

P1, P2, P3, P4, P5, P6, P7, EGO = PlayerName("P1"), PlayerName("P2"), PlayerName("P3"), PlayerName("P4"), PlayerName(
    "P5"), PlayerName("P6"), PlayerName("P7"), PlayerName("Ego")


def get_scenario_bicycle() -> SimContext:
    scenario_name = "USA_Lanker-1_1_T-1"
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name)

    x0_p1 = VehicleStateDyn(x=0, y=0, theta=deg2rad(60), vx=kmh2ms(30), delta=0)
    x0_p2 = VehicleStateDyn(x=24, y=6, theta=deg2rad(150), vx=kmh2ms(20), delta=0)
    x0_p3 = VehicleStateDyn(x=30, y=10, theta=deg2rad(160), vx=kmh2ms(20), delta=0)
    x0_p4 = VehicleStateDyn(x=-4, y=0, theta=deg2rad(60), vx=kmh2ms(30), delta=0)
    x0_p5 = VehicleStateDyn(x=20, y=9, theta=deg2rad(150), vx=kmh2ms(25), delta=0)

    x0_ego = VehicleStateDyn(x=3, y=0, theta=deg2rad(60), vx=kmh2ms(50), delta=0)
    vg_ego = VehicleGeometry.default_car(color="firebrick")
    ego_model = VehicleModelDyn.default_car(x0_ego)
    ego_model.vg = vg_ego

    models = {P1: VehicleModelDyn.default_car(x0_p1),
              P2: VehicleModelDyn.default_bicycle(x0_p2),
              P3: VehicleModelDyn.default_car(x0_p3),
              P4: VehicleModelDyn.default_car(x0_p4),
              P5: VehicleModelDyn.default_bicycle(x0_p5),
              EGO: ego_model
              }

    net = scenario.lanelet_network
    agents: List[LFAgent] = []
    for agent in models:
        if not models[agent].model_type == 'pedestrian':
            x0 = models[agent].get_state()
            p = np.array([x0.x, x0.y])
            lane_id = net.find_lanelet_by_position([p, ])
            assert len(lane_id[0]) > 0, p
            lane = net.find_lanelet_by_id(lane_id[0][0])
            merged_lane = Lanelet.all_lanelets_by_merging_successors_from_lanelet(
                lanelet=lane, network=net)[0][0]
            dglane = DgLanelet.from_commonroad_lanelet(merged_lane)
            sp_controller_param: SpeedControllerParam = SpeedControllerParam(setpoint_minmax=models[agent].vp.vx_limits,
                                                                             output_minmax=models[agent].vp.acc_limits,
                                                                             )
            st_controller_param: SteerControllerParam = SteerControllerParam(setpoint_minmax=(
                                                                                -models[agent].vp.delta_max,
                                                                                models[agent].vp.delta_max),
                                                                             output_minmax=(
                                                                                 -models[agent].vp.ddelta_max,
                                                                                 models[agent].vp.ddelta_max),
                                                                             )
            sp_controller = SpeedController(sp_controller_param)
            st_controller = SteerController(st_controller_param)
            agents.append(LFAgent(dglane, speed_controller=sp_controller, steer_controller=st_controller))
    players = {P1: agents[0],
               P2: agents[1],
               P3: agents[2],
               P4: agents[3],
               P5: agents[4],
               EGO: agents[5], }
    return SimContext(scenario=scenario,
                      models=models,
                      players=players,
                      param=SimParameters(
                          dt=D("0.01"), dt_commands=D("0.1"), sim_time_after_collision=D(4), max_sim_time=D(5)),
                      )


def get_scenario_illegal_turn() -> SimContext:
    sim_param = SimParameters(dt=SimTime("0.01"),
                              dt_commands=SimTime("0.05"),
                              max_sim_time=SimTime(6),
                              sim_time_after_collision=SimTime(6))
    # initialize all contexts/ agents and simulator
    return get_scenario_commonroad_replica(
        scenario_name="USA_Lanker-1_1_T-1.xml", sim_param=sim_param)


def get_merging() -> SimContext:
    sim_param = SimParameters(dt=SimTime("0.01"),
                              dt_commands=SimTime("0.05"),
                              max_sim_time=SimTime(6),
                              sim_time_after_collision=SimTime(6))
    # initialize all contexts/ agents and simulator
    return get_scenario_commonroad_replica(
        scenario_name="ZAM_Zip-1_66_T-1", sim_param=sim_param)


def get_scenario_suicidal_pedestrian() -> SimContext:
    scenario_name = "USA_Peach-1_1_T-1"
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name)
    scenario.translate_rotate(translation=np.array([0, 0]), angle=-pi / 2)
    x0_p3 = PedestrianState(x=-15, y=-18, theta=deg2rad(90), vx=0)
    x0_p1 = VehicleStateDyn(x=-37, y=-8, theta=0.05, vx=kmh2ms(40), delta=0)
    x0_p2 = VehicleStateDyn(x=-35.5, y=-11, theta=0.05, vx=kmh2ms(40), delta=0)
    x0_p4 = VehicleStateDyn(x=-37, y=-14, theta=0.05, vx=kmh2ms(30), delta=0)
    x0_p5 = VehicleStateDyn(x=-10, y=-4, theta=deg2rad(188), vx=kmh2ms(30), delta=0)

    x0_ego = VehicleStateDyn(x=x0_p2.x - 8, y=x0_p2.y, theta=0.00, vx=kmh2ms(50), delta=0)
    vg_ego = VehicleGeometry.default_car(color="firebrick")
    ego_model = VehicleModelDyn.default_car(x0_ego)
    ego_model.vg = vg_ego

    models = {P1: VehicleModelDyn.default_car(x0_p1),
              P2: VehicleModelDyn.default_car(x0_p2),
              P3: PedestrianModel.default(x0_p3),
              P4: VehicleModelDyn.default_car(x0_p4),
              P5: VehicleModelDyn.default_car(x0_p5),
              EGO: ego_model
              }

    ped_commands_plan = DgSampledSequence[PedestrianCommands](
        timestamps=[0, 0.2, 1], values=[PedestrianCommands(acc=0, dtheta=0),
                                        PedestrianCommands(acc=5, dtheta=0.1),
                                        PedestrianCommands(acc=3, dtheta=0)])

    net = scenario.lanelet_network
    agents: List[LFAgent] = []
    for agent in models:
        if not models[agent].model_type == 'pedestrian':
            x0 = models[agent].get_state()
            p = np.array([x0.x, x0.y])
            lane_id = net.find_lanelet_by_position([p, ])
            assert len(lane_id[0]) > 0, p
            lane = net.find_lanelet_by_id(lane_id[0][0])
            merged_lane = Lanelet.all_lanelets_by_merging_successors_from_lanelet(
                lanelet=lane, network=net)[0][0]
            dglane = DgLanelet.from_commonroad_lanelet(merged_lane)
            sp_controller_param: SpeedControllerParam = SpeedControllerParam(setpoint_minmax=models[agent].vp.vx_limits,
                                                                             output_minmax=models[agent].vp.acc_limits,
                                                                             )
            st_controller_param: SteerControllerParam = SteerControllerParam(setpoint_minmax=(
                                                                             -models[agent].vp.delta_max,
                                                                             models[agent].vp.delta_max),
                                                                             output_minmax=(
                                                                             -models[agent].vp.ddelta_max,
                                                                             models[agent].vp.ddelta_max),
                                                                             )
            sp_controller = SpeedController(sp_controller_param)
            st_controller = SteerController(st_controller_param)
            agents.append(LFAgent(dglane, speed_controller=sp_controller, steer_controller=st_controller))
    #
    # plt.savefig("lanes_debug.png", dpi=300)
    #
    players = {P1: agents[0],
               P2: agents[1],
               P3: NPAgent(ped_commands_plan),
               P4: agents[2],
               P5: agents[3],
               EGO: agents[4],
               }

    return SimContext(scenario=scenario,
                      models=models,
                      players=players,
                      param=SimParameters(
                          dt=D("0.01"), dt_commands=D("0.1"), sim_time_after_collision=D(6), max_sim_time=D(7)),
                      )


def get_two_lanes_scenario() -> SimContext:
    scenario_name = "ZAM_Zip-1_66_T-1"
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name)

    x0_truck = VehicleStateDyn(x=-98, y=5.35, theta=0.00, vx=kmh2ms(30), delta=0)
    x0_p2 = VehicleStateDyn(x=-105, y=9, theta=0.00, vx=kmh2ms(60), delta=0)
    x0_ego = VehicleStateDyn(x=-115, y=5.3, theta=0.00, vx=kmh2ms(90), delta=0)

    truck_model = VehicleModelDyn.default_truck(x0_truck)
    vg_ego = VehicleGeometry.default_car(color="firebrick")
    ego_model = VehicleModelDyn.default_car(x0_ego)
    ego_model.vg = vg_ego

    models = {P1: truck_model,
              P2: VehicleModelDyn.default_car(x0_p2),
              EGO: ego_model
              }

    net = scenario.lanelet_network
    agents: List[LFAgent] = []
    for agent in models:
        if not models[agent].model_type == 'pedestrian':
            x0 = models[agent].get_state()
            p = np.array([x0.x, x0.y])
            lane_id = net.find_lanelet_by_position([p, ])
            assert len(lane_id[0]) > 0, p
            lane = net.find_lanelet_by_id(lane_id[0][0])
            merged_lane = Lanelet.all_lanelets_by_merging_successors_from_lanelet(
                lanelet=lane, network=net)[0][0]
            dglane = DgLanelet.from_commonroad_lanelet(merged_lane)
            sp_controller_param: SpeedControllerParam = SpeedControllerParam(setpoint_minmax=models[agent].vp.vx_limits,
                                                                             output_minmax=models[agent].vp.acc_limits,
                                                                             )
            st_controller_param: SteerControllerParam = SteerControllerParam(setpoint_minmax=(
                                                                                 -models[agent].vp.delta_max,
                                                                                 models[agent].vp.delta_max),
                                                                             output_minmax=(
                                                                                 -models[agent].vp.ddelta_max,
                                                                                 models[agent].vp.ddelta_max),
                                                                             )
            sp_controller = SpeedController(sp_controller_param)
            st_controller = SteerController(st_controller_param)
            agents.append(LFAgent(dglane, speed_controller=sp_controller, steer_controller=st_controller))
    players = {P1: agents[0],
               P2: agents[1],
               EGO: agents[2],
               }

    return SimContext(scenario=scenario,
                      models=models,
                      players=players,
                      param=SimParameters(
                          dt=D("0.01"), dt_commands=D("0.1"), sim_time_after_collision=D(6), max_sim_time=D(7)),
                      )


def get_scenario_racetrack_test() -> SimContext:
    scenario_name = "DEU_Hhr-1_1"
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name)
    lane = scenario.lanelet_network.lanelets[0]
    lane = Lanelet.all_lanelets_by_merging_successors_from_lanelet(lane, scenario.lanelet_network, max_length=1000)[0][
        0]
    dglane = DgLanelet.from_commonroad_lanelet(lane)

    start = dglane.center_point(10)
    xytheta = xytheta_from_SE2(start)
    x0_p1 = VehicleStateDyn(x=xytheta[0], y=xytheta[1], theta=xytheta[2], vx=kmh2ms(50), delta=0)

    models = {P1: VehicleModelDyn.default_car(x0_p1)}
    players = {P1: LFAgent(dglane)}

    return SimContext(scenario=scenario,
                      models=models,
                      players=players,
                      param=SimParameters(dt=D("0.01"), sim_time_after_collision=D(3), max_sim_time=D(10)),
                      )
