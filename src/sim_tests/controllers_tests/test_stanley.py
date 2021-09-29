from dg_commons.controllers.speed import SpeedController, SpeedControllerParam, SpeedBehaviorParam
from dg_commons.controllers.stanley_controller import StanleyParam, Stanley
from dg_commons.controllers.steering_controllers import SCP, SCPParam
from dg_commons.analysis.metrics import DeviationLateral, DeviationVelocity, SteeringVelocity, Acceleration
from sim.agents.lane_followers import LFAgentStanley
from dg_commons.state_estimators.extended_kalman_filter import ExtendedKalman, ExtendedKalmanParam
from sim_tests.controllers_tests.test_controller_utils import run_test
from dg_commons.controllers.full_controller_base import VehicleController
from dg_commons.utils import SemiDef
from dg_commons.state_estimators.dropping_trechniques import *


def test_stanley(scenario):
    controller = VehicleController(

        controller=Stanley,
        controller_params=StanleyParam(
            stanley_gain=2
        ),

        lf_agent=LFAgentStanley,

        longitudinal_controller=SpeedController,
        longitudinal_controller_params=SpeedControllerParam(
            kP=1,
            kI=0.01,
            kD=0.1
        ),

        speed_behavior_param=SpeedBehaviorParam(
            nominal_speed=8
        ),

        steering_controller=SCP,
        steering_controller_params=SCPParam(
            ddelta_kp=10
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


# test_stanley()
