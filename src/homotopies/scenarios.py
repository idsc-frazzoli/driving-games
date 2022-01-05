from numpy import deg2rad
from decimal import Decimal as D
from dg_commons import PlayerName, DgSampledSequence
from dg_commons.sim import SimParameters
from dg_commons.sim.agents import NPAgent
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.sim.models.vehicle_dynamic import VehicleStateDyn, VehicleModelDyn
from dg_commons.sim.scenarios import load_commonroad_scenario
from dg_commons.sim.simulator import SimContext
from homotopies.NonlinearCons.mpc_agent import MpcAgent
from homotopies.NonlinearCons.mpcc_agent import MpccAgent
from homotopies.MIP.MIP_agent import MIPAgent

P1, P2 = (
    PlayerName("P1"),
    PlayerName("P2"),
)


# basic test for lane following scenario(orientations of ego-vehicle and surrounding are roughly parallel)
def get_homotopy_scenario(agent_type: str) -> SimContext:
    scenario_name = "USA_Lanker-1_1_T-1"
    scenario_dir = "scenarios"
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name, scenario_dir)

    x0_p1 = VehicleStateDyn(x=0, y=0, theta=deg2rad(60), vx=5, delta=0)
    x0_p2 = VehicleStateDyn(x=12, y=20, theta=deg2rad(60), vx=0, delta=0)
    models = {P1: VehicleModelDyn.default_car(x0_p1), P2: VehicleModelDyn.default_car(x0_p2)}

    static_vehicle = DgSampledSequence[VehicleCommands](
        timestamps=[
            0,
        ],
        values=[
            VehicleCommands(acc=0, ddelta=0),
        ],
    )
    target_pos = [20, 35]
    ref_path = [[0, 0], [20, 35]]
    if agent_type == 'mpc':
        control_agent = MpcAgent(target_pos)
    elif agent_type == 'mpcc':
        control_agent = MpccAgent(ref_path)
    elif agent_type == 'MIP':
        control_agent = MIPAgent(ref_path)
    else:
        control_agent = NPAgent(static_vehicle)
    players = {P1: control_agent, P2: NPAgent(static_vehicle)}

    return SimContext(
        scenario=scenario,
        models=models,
        players=players,
        param=SimParameters(dt=D("0.01"), dt_commands=D("0.1"), sim_time_after_collision=D(2), max_sim_time=D(7)),
    )


# basic test for lane following scenario(orientations of ego-vehicle and surrounding are roughly perpendicular)
def get_intersection_scenario(agent_type: str) -> SimContext:
    scenario_name = "USA_Lanker-1_1_T-1"
    scenario_dir = "scenarios"
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name, scenario_dir)

    x0_p1 = VehicleStateDyn(x=0, y=0, theta=deg2rad(60), vx=5, delta=0)
    #x0_p2 = VehicleStateDyn(x=10, y=20, theta=deg2rad(160), vx=0, delta=0)
    #x0_p2: VehicleStateDyn = VehicleStateDyn(x=15, y=15, theta=deg2rad(155), vx=2.5, delta=0)
    x0_p2: VehicleStateDyn = VehicleStateDyn(x=20, y=12.5, theta=deg2rad(155), vx=2.5, delta=0)
    models = {P1: VehicleModelDyn.default_car(x0_p1), P2: VehicleModelDyn.default_car(x0_p2)}

    static_vehicle = DgSampledSequence[VehicleCommands](
        timestamps=[
            0,
        ],
        values=[
            VehicleCommands(acc=0, ddelta=0),
        ],
    )
    target_pos = [14, 34]
    ref_path = [[0, 0], [17, 35]]
    if agent_type == 'mpc':
        control_agent = MpcAgent(target_pos)
    elif agent_type == 'mpcc':
        control_agent = MpccAgent(ref_path)
    elif agent_type == 'MIP':
        control_agent = MIPAgent(ref_path)
    else:
        control_agent = NPAgent(static_vehicle)
    players = {P1: control_agent, P2: NPAgent(static_vehicle)}

    return SimContext(
        scenario=scenario,
        models=models,
        players=players,
        param=SimParameters(dt=D("0.01"), dt_commands=D("0.1"), sim_time_after_collision=D(2), max_sim_time=D(7)),
    )
