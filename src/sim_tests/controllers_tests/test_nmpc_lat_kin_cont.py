from sim_tests.controllers_tests.test_controller import DT_COMMANDS
from dg_commons.controllers.speed import SpeedController, SpeedControllerParam, SpeedBehaviorParam
from dg_commons.controllers.mpc.nmpc_lateral_kin_cont import NMPCLatKinContPVParam, NMPCLatKinContPV
from dg_commons.analysis.metrics import DeviationLateral, DeviationVelocity, Acceleration, SteeringVelocity
from sim.agents.lane_followers import LFAgentLatMPC
from dg_commons.state_estimators.extended_kalman_filter import ExtendedKalman, ExtendedKalmanParam
from sim_tests.controllers_tests.test_controller_utils import run_test
from dg_commons.controllers.full_controller_base import VehicleController
from dg_commons.controllers.mpc.mpc_utils.cost_functions import *
from dg_commons.state_estimators.dropping_trechniques import *


def test_nmpc_lat_kin_cont():
    scenario = "USA_Peach-1_1_T-1"
    # scenario="ZAM_Tjunction-1_129_T-1"
    # scenario="ARG_Carcarana-1_1_T-1"

    controller = VehicleController(

        controller=NMPCLatKinContPV,
        controller_params=NMPCLatKinContPVParam(
            n_horizon=15,
            t_step=float(DT_COMMANDS),
            cost="quadratic",
            cost_params=QuadraticParams(
                q=SemiDef(matrix=np.eye(2)),
                r=SemiDef(matrix=np.eye(1))
            ),
            delta_input_weight=1e-2,
            path_approx_technique='linear',
            rear_axle=False
        ),

        lf_agent=LFAgentLatMPC,

        longitudinal_controller=SpeedController,
        longitudinal_controller_params=SpeedControllerParam(
            kP=1,
            kI=0.01,
            kD=0.1
        ),

        speed_behavior_param=SpeedBehaviorParam(
            nominal_speed=8
        ),

        metrics=[
            DeviationLateral,
            DeviationVelocity,
            Acceleration,
            SteeringVelocity
        ],

        state_estimator=ExtendedKalman,
        state_estimator_params=ExtendedKalmanParam(
            actual_model_var=SemiDef([i*1 for i in [0.0001, 0.0001, 0.0001, 0.0001, 0.0001]]),
            actual_meas_var=SemiDef([i*0 for i in [0.001, 0.001, 0.001, 0.001, 0.001]]),
            belief_model_var=SemiDef([i*1 for i in [0.0001, 0.0001, 0.0001, 0.0001, 0.0001]]),
            belief_meas_var=SemiDef([i*0 for i in [0.001, 0.001, 0.001, 0.001, 0.001]]),
            initial_variance=SemiDef(matrix=np.zeros((5, 5))),
            dropping_technique=LGB,
            dropping_params=LGBParam(
                failure_p=0.0
            )
        )
    )

    run_test(controller, scenario)


# test_nmpc_lat_kin_cont()
