from sim_tests.controllers_tests.test_controller import DT_COMMANDS
from dg_commons.controllers.speed import SpeedController, SpeedControllerParam, SpeedBehaviorParam
from dg_commons.controllers.mpc.nmpc_lateral_kin_dis import NMPCLatKinDisPVParam, NMPCLatKinDisPV
from dg_commons.analysis.metrics import DeviationLateral, DeviationVelocity
from sim.agents.lane_followers import LFAgentLatMPC
from dg_commons.state_estimators.extended_kalman_filter import ExtendedKalman, ExtendedKalmanParam
import numpy as np
from sim_tests.controllers_tests.test_controller_utils import run_test
from dg_commons.controllers.full_controller_base import VehicleController


def test_nmpc_lat_kin_dis():
    scenario = "USA_Peach-1_1_T-1"
    # scenario="ZAM_Tjunction-1_129_T-1"
    # scenario="ARG_Carcarana-1_1_T-1"

    controller = VehicleController(

        controller=NMPCLatKinDisPV,
        controller_params=NMPCLatKinDisPVParam(
            n_horizon=15,
            t_step=float(DT_COMMANDS),
            position_err_weight=1,
            steering_vel_weight=1,
            delta_input_weight=1e-2,
            dis_technique='Kinematic RK4',  # 'Kinematic Euler' or 'Kinematic RK4' or 'Anstrom Euler'
            path_approx_technique='linear',
            dis_t=0.05
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
            DeviationVelocity
        ],

        state_estimator=ExtendedKalman,
        state_estimator_params=ExtendedKalmanParam(
            actual_model_var=0.0001 * np.eye(5),
            actual_meas_var=0.001 * np.eye(5) * 0,
            belief_model_var=0.0001 * np.eye(5),
            belief_meas_var=0.001 * np.eye(5) * 0
        )
    )

    run_test(controller, scenario)


# test_nmpc_lat_kin_dis()
