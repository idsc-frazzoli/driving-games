from sim_tests.controllers_tests.test_controller import TestController, DT_COMMANDS
from dg_commons.controllers.speed import SpeedBehavior, SpeedBehaviorParam
from dg_commons.controllers.mpc.nmpc_full_kin_dis import NMPCFullKinDisPVParam, NMPCFullKinDisPV
from dg_commons.controllers.steering_controllers import SCIdentityParam, SCIdentity
from dg_commons.analysis.metrics import DeviationLateral, DeviationVelocity
from sim.agents.lane_followers import LFAgentFullMPC
from dg_commons.state_estimators.extended_kalman_filter import ExtendedKalman, ExtendedKalmenParam
import numpy as np


def test_mpckin():
    scenario_name: str = "USA_Peach-1_1_T-1"
    # scenario_name: str = "ZAM_Tjunction-1_129_T-1"
    # scenario_name: str = "ARG_Carcarana-1_1_T-1"
    """Name of the chosen scenario"""
    vehicle_speed: float = 8
    """Nominal speed of the vehicle"""
    n_horizon = 15
    """ Horizon Length """
    t_step = float(DT_COMMANDS)
    """ Sample Time """
    state_mult = 1
    """ Weighting factor in cost function for having state error """
    input_mult = 1
    """ Weighting factor in cost function for applying input """
    speed_mult: float = 1
    """ Weighting factor in cost function for velocity error """
    acc_mult: float = 1
    """ Weighting factor in cost function for acceleration """
    delta_input_mult = 1e-2
    """ Weighting factor in cost function for varying input """
    dis_technique: str = 'Kinematic RK4'  # 'Kinematic Euler' or 'Kinematic RK4'
    """ Discretization technique """
    dis_t = 0.05
    """ Discretization interval """
    modeling_variance: np.ndarray = 0.0001*np.eye(5)
    """ Modeling variance matrix """
    measurement_variance: np.ndarray = 0.001*np.eye(5)
    """ Measurement variance matrix """
    belief_modeling_variance: np.ndarray = 0.0001*np.eye(5)
    """ Modeling variance matrix """
    belief_measurement_variance: np.ndarray = 0.001*np.eye(5)
    """ Measurement variance matrix """

    sp_behavior_param: SpeedBehaviorParam = SpeedBehaviorParam(nominal_speed=vehicle_speed)
    sp_behavior = {"Name": "Speed Behavior", "Behavior": SpeedBehavior, "Parameters": sp_behavior_param}
    """Speed behavior"""
    mpc_param: NMPCFullKinDisPVParam = NMPCFullKinDisPVParam(n_horizon=n_horizon, t_step=t_step, state_mult=state_mult,
                                                             input_mult=input_mult, delta_input_mult=delta_input_mult,
                                                             speed_mult=speed_mult, dis_technique=dis_technique,
                                                             dis_t=dis_t, acc_mult=acc_mult)
    mpc_controller = {"Name": "MPC Controller", "Controller": NMPCFullKinDisPV, "Parameters": mpc_param}
    """MPC Controller"""
    steering_param: SCIdentityParam = SCIdentityParam()
    steering_controller = {"Name": "Identity controller", "Controller": SCIdentity, "Parameters": steering_param}
    """Pure Pursuit Controller"""
    metrics = [DeviationLateral, DeviationVelocity]
    """Metrics"""
    state_estimator_params: ExtendedKalmenParam = ExtendedKalmenParam(actual_meas_var=measurement_variance,
                                                                      actual_model_var=modeling_variance,
                                                                      belief_meas_var=belief_measurement_variance,
                                                                      belief_model_var=belief_modeling_variance
                                                                      )
    state_estimator = {"Name": "Extended Kalman", "Estimator": ExtendedKalman, "Parameters": state_estimator_params}
    """ State Estimator """

    test_pp = TestController(scenario_name, "-", metrics, LFAgentFullMPC,
                             mpc_controller, sp_behavior, steering_controller, state_estimator=state_estimator)
    test_pp.run()
    test_pp.evaluate_metrics()
    test_pp.evaluate_metrics_test()
    test_pp.to_json()


test_mpckin()
