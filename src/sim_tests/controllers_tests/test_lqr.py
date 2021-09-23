from dg_commons.analysis.metrics import DeviationLateral, DeviationVelocity
from dg_commons.controllers.speed import SpeedController, SpeedControllerParam, SpeedBehaviorParam
from dg_commons.controllers.lqr import LQRParam, LQR
from dg_commons.controllers.steering_controllers import SCP, SCPParam
from sim.agents.lane_followers import LFAgentLQR
from dg_commons.state_estimators.extended_kalman_filter import ExtendedKalman, ExtendedKalmanParam
import numpy as np
from sim_tests.controllers_tests.test_controller_utils import run_test
from dg_commons.controllers.full_controller_base import VehicleController
from dg_commons.utils import SemiDef


def test_lqr():
    scenario = "USA_Peach-1_1_T-1"
    # scenario="ZAM_Tjunction-1_129_T-1"
    # scenario="ARG_Carcarana-1_1_T-1"

    controller = VehicleController(

        controller=LQR,
        controller_params=LQRParam(
            r=SemiDef([0.8]),
            q=SemiDef(matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.01]]))
        ),

        lf_agent=LFAgentLQR,

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
            actual_model_var=SemiDef([i*1 for i in [0.0001, 0.0001, 0.0001, 0.0001, 0.0001]]),
            actual_meas_var=SemiDef([i*0 for i in [0.001, 0.001, 0.001, 0.001, 0.001]]),
            belief_model_var=SemiDef([i*1 for i in [0.0001, 0.0001, 0.0001, 0.0001, 0.0001]]),
            belief_meas_var=SemiDef([i*0 for i in [0.001, 0.001, 0.001, 0.001, 0.001]])
        )
    )

    run_test(controller, scenario)


# test_lqr()
