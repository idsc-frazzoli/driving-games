from numpy import deg2rad
from decimal import Decimal as D
from dg_commons import PlayerName, DgSampledSequence
from dg_commons.sim import SimParameters
from dg_commons.sim.agents import NPAgent
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.sim.models.vehicle_dynamic import VehicleStateDyn, VehicleModelDyn
from dg_commons.sim.scenarios import load_commonroad_scenario
from dg_commons.sim.simulator import SimContext

P1, P2 = (
    PlayerName("P1"),
    PlayerName("P2"),
)


def get_homotopy_scenario() -> SimContext:
    scenario_name = "USA_Lanker-1_1_T-1"
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name)

    x0_p1 = VehicleStateDyn(x=0, y=0, theta=deg2rad(60), vx=5, delta=0)
    x0_p2 = VehicleStateDyn(x=24, y=6, theta=deg2rad(150), vx=6, delta=0)
    models = {P1: VehicleModelDyn.default_car(x0_p1), P2: VehicleModelDyn.default_bicycle(x0_p2)}

    static_vehicle = DgSampledSequence[VehicleCommands](
        timestamps=[
            0,
        ],
        values=[
            VehicleCommands(acc=0, ddelta=0),
        ],
    )

    players = {P1: agents[0], P2: NPAgent(static_vehicle)}
    return SimContext(
        scenario=scenario,
        models=models,
        players=players,
        param=SimParameters(dt=D("0.01"), dt_commands=D("0.1"), sim_time_after_collision=D(6), max_sim_time=D(7)),
    )
