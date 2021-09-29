from sim_tests.controllers_tests.test_controller import DT_COMMANDS
from dg_commons.controllers.speed import SpeedBehaviorParam
from dg_commons.controllers.mpc.nmpc_full_kin_cont import NMPCFullKinContAN, NMPCFullKinContANParam
from dg_commons.analysis.metrics import DeviationLateral, DeviationVelocity, SteeringVelocity, Acceleration
from sim.agents.lane_followers import LFAgentFullMPC
from dg_commons.state_estimators.extended_kalman_filter import ExtendedKalman, ExtendedKalmanParam
from sim_tests.controllers_tests.test_controller_utils import run_test
from dg_commons.controllers.full_controller_base import VehicleController
from dg_commons.controllers.mpc.mpc_utils.cost_functions import *
from dg_commons.state_estimators.dropping_trechniques import *


def test_nmpc_full_kin_analytical(scenario):
    controller = VehicleController(

        controller=NMPCFullKinContAN,
        controller_params=NMPCFullKinContANParam(
            n_horizon=15,
            t_step=float(DT_COMMANDS),
            cost="quadratic",
            cost_params=QuadraticParams(
                q=SemiDef(matrix=np.eye(3)),
                r=SemiDef(matrix=np.eye(2))
            ),
            delta_input_weight=1e-2,
            path_approx_technique='linear',
            rear_axle=False
        ),

        lf_agent=LFAgentFullMPC,

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


# test_nmpc_full_kin_analytical()
