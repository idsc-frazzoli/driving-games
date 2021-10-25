from decimal import Decimal as D
from math import pi
from typing import List
import os

import numpy as np
from commonroad.scenario.lanelet import Lanelet
from geometry import xytheta_from_SE2
from numpy import deg2rad, linspace

from crash.agents import B1Agent, B2Agent
from crash.agents.pred_agent import PredAgent
from dg_commons import DgSampledSequence, PlayerName
from dg_commons.controllers.speed import SpeedControllerParam, SpeedController
from dg_commons.controllers.steer import SteerControllerParam, SteerController
from dg_commons.maps.lanes import DgLanelet
from dg_commons.sim import SimTime
from dg_commons.sim.agents.agent import NPAgent, Agent
from dg_commons.sim.models import kmh2ms, PEDESTRIAN
from dg_commons.sim.models.pedestrian import PedestrianState, PedestrianModel, PedestrianCommands
from dg_commons.sim.models.vehicle_dynamic import VehicleStateDyn, VehicleModelDyn
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from sim.scenarios.utils import load_commonroad_scenario
from dg_commons.sim.scenarios.agent_from_commonroad import dglane_from_position
from dg_commons.sim.scenarios.factory import get_scenario_commonroad_replica
from dg_commons.sim.simulator import SimContext
from dg_commons.sim.simulator_structures import SimParameters

__all__ = ["get_scenario_bicycles", "get_scenario_illegal_turn", "get_scenario_suicidal_pedestrian",
           "get_scenario_two_lanes", "get_scenario_racetrack_test", "get_scenario_predictions"]

P1, P2, P3, P4, P5, P6, P7, EGO = PlayerName("P1"), PlayerName("P2"), PlayerName("P3"), PlayerName("P4"), PlayerName(
    "P5"), PlayerName("P6"), PlayerName("P7"), PlayerName("Ego")


def get_scenario_bicycles() -> SimContext:
    scenario_name = "USA_Lanker-1_1_T-1"
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name)

    x0_p1 = VehicleStateDyn(x=0, y=0, theta=deg2rad(60), vx=5, delta=0)
    x0_p2 = VehicleStateDyn(x=24, y=6, theta=deg2rad(150), vx=6, delta=0)
    x0_p3 = VehicleStateDyn(x=30, y=10, theta=deg2rad(170), vx=7, delta=0)
    x0_p4 = VehicleStateDyn(x=-4, y=0, theta=deg2rad(60), vx=5, delta=0)
    x0_p5 = VehicleStateDyn(x=22, y=6, theta=deg2rad(150), vx=8, delta=0)

    x0_ego = VehicleStateDyn(x=2.5, y=-3, theta=deg2rad(60), vx=kmh2ms(50), delta=0)
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
    agents: List[B1Agent] = []
    for pname in models:
        assert not models[pname].model_type == PEDESTRIAN
        x0 = models[pname].get_state()
        p = np.array([x0.x, x0.y])
        dglane = dglane_from_position(p, net)
        sp_controller_param: SpeedControllerParam = SpeedControllerParam(
            setpoint_minmax=models[pname].vp.vx_limits, output_minmax=models[pname].vp.acc_limits)
        st_controller_param: SteerControllerParam = SteerControllerParam(
            setpoint_minmax=(-models[pname].vp.delta_max, models[pname].vp.delta_max),
            output_minmax=(-models[pname].vp.ddelta_max, models[pname].vp.ddelta_max))
        sp_controller = SpeedController(sp_controller_param)
        st_controller = SteerController(st_controller_param)
        agents.append(B1Agent(dglane, speed_controller=sp_controller, steer_controller=st_controller))

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
    sim_context = get_scenario_commonroad_replica(
        scenario_name="USA_Lanker-1_1_T-1.xml", sim_param=sim_param, ego_player=PlayerName("P16"))

    return sim_context


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
    agents: List[B1Agent] = []
    for x0 in [x0_p1, x0_p2, x0_p4, x0_p5, x0_ego]:
        p = np.array([x0.x, x0.y])
        dglane = dglane_from_position(p, net)
        # ####### debug
        # points = dglane.lane_profile()
        # xp, yp = zip(*points)
        # x = np.array(xp)
        # y = np.array(yp)
        # plt.fill(x, y, alpha=0.2, zorder=15)
        # #######
        agents.append(B1Agent(dglane))
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


def get_scenario_two_lanes() -> SimContext:
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
    agents: List[B2Agent] = []
    for agent in models:
        if not models[agent].model_type == 'pedestrian':
            x0 = models[agent].get_state()
            p = np.array([x0.x, x0.y])
            dglane = dglane_from_position(p, net)
            sp_controller_param: SpeedControllerParam = SpeedControllerParam(
                setpoint_minmax=models[agent].vp.vx_limits, output_minmax=models[agent].vp.acc_limits, )
            st_controller_param: SteerControllerParam = SteerControllerParam(
                setpoint_minmax=(-models[agent].vp.delta_max, models[agent].vp.delta_max),
                output_minmax=(-models[agent].vp.ddelta_max, models[agent].vp.ddelta_max), )
            sp_controller = SpeedController(sp_controller_param)
            st_controller = SteerController(st_controller_param)
            agents.append(B2Agent(dglane, speed_controller=sp_controller, steer_controller=st_controller))
    players = {P1: agents[0],
               P2: agents[1],
               EGO: agents[2],
               }

    return SimContext(scenario=scenario,
                      models=models,
                      players=players,
                      param=SimParameters(
                          dt=D("0.01"), dt_commands=D("0.1"), sim_time_after_collision=D(4), max_sim_time=D(6)),
                      )


def get_scenario_racetrack_test() -> SimContext:
    scenario_name = "DEU_Hhr-1_1"
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name)
    lane = scenario.lanelet_network.lanelets[0]
    lane = Lanelet.all_lanelets_by_merging_successors_from_lanelet(lane, scenario.lanelet_network, max_length=1000)[0][
        0]
    dglane = DgLanelet.from_commonroad_lanelet(lane)

    betas = linspace(-1, 5, 500).tolist()
    # plt.figure()
    # for beta in betas:
    #     q = dglane.center_point(beta)
    #     radius = dglane.radius(beta)
    #     delta_left = np.array([0, radius])
    #     delta_right = np.array([0, -radius])
    #     left = SE2_apply_R2(q, delta_left)
    #     right = SE2_apply_R2(q, delta_right)
    #     plt.plot(*left, "o")
    #     plt.plot(*right, "x")
    #     plt.gca().set_aspect("equal")
    # plt.savefig(f"out/debug{lane.lanelet_id}.png")
    # plt.close()

    start = dglane.center_point(10)
    xytheta = xytheta_from_SE2(start)
    x0_p1 = VehicleStateDyn(x=xytheta[0], y=xytheta[1], theta=xytheta[2], vx=kmh2ms(50), delta=0)

    models = {P1: VehicleModelDyn.default_car(x0_p1)}
    players = {P1: B1Agent(dglane)}

    return SimContext(scenario=scenario,
                      models=models,
                      players=players,
                      param=SimParameters(dt=D("0.01"), sim_time_after_collision=D(3), max_sim_time=D(10)),
                      )


def get_scenario_predictions() -> SimContext:
    scenario_name = "ZAM_Zip-1_66_T-1"
    # question: planning problem set is never used. How come?
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name)

    x0_p1 = VehicleStateDyn(x=-98, y=5.35, theta=0.00, vx=24.5, delta=0)
    x0_ego = VehicleStateDyn(x=-115, y=9, theta=0.00, vx=kmh2ms(90), delta=0)
    ego_model = VehicleModelDyn.default_car(x0_ego)
    ego_model.vg = VehicleGeometry.default_car(color="firebrick")

    models = {P1: VehicleModelDyn.default_car(x0_p1),
              EGO: ego_model
              }

    net = scenario.lanelet_network
    agents: List[Agent] = [] # todo: is [Agent] instead of [B1Agent] correct?

    for pname in models:
        assert not models[pname].model_type == PEDESTRIAN # question: why do we need to check there are no pedestrians?
        x0 = models[pname].get_state()
        p = np.array([x0.x, x0.y])
        dglane = dglane_from_position(p, net) # find lane that controller will follow. Lane is a list of LaneCtrlPoint

        # instantiate speed and steering controllers and their paramenters
        sp_controller_param: SpeedControllerParam = SpeedControllerParam(
            setpoint_minmax=models[pname].vp.vx_limits, output_minmax=models[pname].vp.acc_limits)
        st_controller_param: SteerControllerParam = SteerControllerParam(
            setpoint_minmax=(-models[pname].vp.delta_max, models[pname].vp.delta_max),
            output_minmax=(-models[pname].vp.ddelta_max, models[pname].vp.ddelta_max))
        sp_controller = SpeedController(sp_controller_param)
        st_controller = SteerController(st_controller_param)

        if pname == EGO:
            agents.append(PredAgent(dglane, speed_controller=sp_controller, steer_controller=st_controller))
        else:
            agents.append(B1Agent(dglane, speed_controller=sp_controller, steer_controller=st_controller))

    players = {P1: agents[0],
               EGO: agents[1], } # question: how does SimContext know that EGO is the one predicting what others do? It doesn't know, right?
    return SimContext(scenario=scenario,
                      models=models,
                      players=players,
                      param=SimParameters(
                          dt=D("0.01"), dt_commands=D("0.1"), sim_time_after_collision=D(4), max_sim_time=D(5)),
                      )