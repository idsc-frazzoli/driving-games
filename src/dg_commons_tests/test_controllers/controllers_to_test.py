from sim.agents.lane_followers import *
from dg_commons.controllers.speed import SpeedController, SpeedControllerParam, SpeedBehaviorParam
from dg_commons.state_estimators.extended_kalman_filter import ExtendedKalman, ExtendedKalmanParam
from dg_commons.controllers.full_controller_base import VehicleController
from dg_commons_tests.test_controllers.controller_test_utils import DT_COMMANDS
from dg_commons.controllers.mpc.mpc_utils.cost_functions import *
from dg_commons.state_estimators.dropping_trechniques import *


TestLQR = VehicleController(
            controller=LQR,
            controller_params=LQRParam(
                r=SemiDef([0.5]),
                q=SemiDef(matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.01]])),
                t_step=float(DT_COMMANDS)
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

            state_estimator=None,
            state_estimator_params=None
        )

TestNMPCFullKinContPV = VehicleController(

        controller=NMPCFullKinContPV,
        lf_agent=LFAgentFullMPC,

        controller_params=NMPCFullKinContPVParam(
            n_horizon=15,
            t_step=float(DT_COMMANDS),
            delta_input_weight=1e-2,
            cost='quadratic',
            cost_params=QuadraticParams(
                q=SemiDef(matrix=np.eye(3)),
                r=SemiDef(matrix=np.eye(2))
            ),
            path_approx_technique='linear',
            rear_axle=False
        ),

        speed_behavior_param=SpeedBehaviorParam(
            nominal_speed=8
        ),

        state_estimator=None,
        state_estimator_params=None
    )


TestNMPCFullKinDisPV = VehicleController(

        controller=NMPCFullKinDisPV,
        controller_params=NMPCFullKinDisPVParam(
            n_horizon=15,
            t_step=float(DT_COMMANDS),
            cost="quadratic",
            cost_params=QuadraticParams(
                q=SemiDef(matrix=np.eye(3)),
                r=SemiDef(matrix=np.eye(2))
            ),
            delta_input_weight=1e-2,
            dis_technique='Kinematic RK4',  # 'Kinematic Euler' or 'Kinematic RK4' or 'Anstrom Euler'
            path_approx_technique='linear',
            dis_t=0.05,
            rear_axle=False
        ),

        lf_agent=LFAgentFullMPC,

        speed_behavior_param=SpeedBehaviorParam(
            nominal_speed=8
        ),

        state_estimator=None,
        state_estimator_params=None
    )

TestNMPCFullKinContAN = VehicleController(

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

        state_estimator=None,
        state_estimator_params=None
    )


TestNMPCLatKinContPV = VehicleController(

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

        state_estimator=None,
        state_estimator_params=None
)

TestNMPCLatKinDisPV = VehicleController(
        controller=NMPCLatKinDisPV,
        controller_params=NMPCLatKinDisPVParam(
            n_horizon=15,
            t_step=float(DT_COMMANDS),
            cost="quadratic",
            cost_params=QuadraticParams(
                q=SemiDef(matrix=np.eye(2)),
                r=SemiDef(matrix=np.eye(1))
            ),
            delta_input_weight=1e-2,
            dis_technique='Kinematic RK4',  # 'Kinematic Euler' or 'Kinematic RK4' or 'Anstrom Euler'
            path_approx_technique='linear',
            dis_t=0.05,
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

        state_estimator=None,
        state_estimator_params=None
    )

TestNMPCLatKinContAN = VehicleController(
        controller=NMPCLatKinContAN,
        controller_params=NMPCLatKinContANParam(
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

        state_estimator=None,
        state_estimator_params=None
    )

TestPurePursuit = VehicleController(

        controller=PurePursuit,
        controller_params=PurePursuitParam(
            k_lookahead=1.8,
            t_step=float(DT_COMMANDS)
        ),

        lf_agent=LFAgentPP,

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

        state_estimator=None,
        state_estimator_params=None
    )


TestStanley = VehicleController(

        controller=Stanley,
        controller_params=StanleyParam(
            stanley_gain=2,
            t_step=float(DT_COMMANDS)
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

        state_estimator=None,
        state_estimator_params=None
    )

contr_to_test = [TestLQR, TestStanley, TestPurePursuit,
                 TestNMPCLatKinContPV, TestNMPCLatKinContAN, TestNMPCFullKinDisPV,
                 TestNMPCLatKinDisPV, TestNMPCFullKinContAN, TestNMPCFullKinContPV]
