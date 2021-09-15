from sim_tests.controllers_tests.test_controller import TestController, DT_COMMANDS
from dg_commons.controllers.speed import SpeedBehavior, SpeedController, SpeedControllerParam, SpeedBehaviorParam
from dg_commons.controllers.mpc.nmpc_lateral_kin_cont import NMPCLatKinContPVParam, NMPCLatKinContPV
from dg_commons.controllers.steering_controllers import SCIdentityParam, SCIdentity
from dg_commons.analysis.metrics import DeviationLateral, DeviationVelocity


def test_mpckin_path_var():
    scenario_name: str = "USA_Peach-1_1_T-1"
    # scenario_name: str = "ZAM_Tjunction-1_129_T-1"
    # scenario_name: str = "ARG_Carcarana-1_1_T-1"
    """Name of the chosen scenario"""
    vehicle_speed: float = 8
    """Nominal speed of the vehicle"""
    speed_kp: float = 1
    """Propotioanl gain longitudinal speed controller"""
    speed_ki: float = 0.01
    """Integral gain longitudinal speed controller"""
    speed_kd: float = 0.1
    """Derivative gain longitudinal speed controller"""
    n_horizon = 15
    """ Horizon Length """
    t_step = float(DT_COMMANDS)
    """ Sample Time """
    state_mult = 1
    """ Weighting factor in cost function for having state error """
    input_mult = 1
    """ Weighting factor in cost function for applying input """
    delta_input_mult = 1e-2
    """ Weighting factor in cost function for varying input """
    technique: str = 'linear'
    """ Path approximation technique """

    sp_controller_param: SpeedControllerParam = SpeedControllerParam(kP=speed_kp, kI=speed_ki, kD=speed_kd)
    sp_controller = {"Name": "Speed Controller", "Controller": SpeedController, "Parameters": sp_controller_param}
    """Speed Controller"""
    sp_behavior_param: SpeedBehaviorParam = SpeedBehaviorParam(nominal_speed=vehicle_speed)
    sp_behavior = {"Name": "Speed Behavior", "Behavior": SpeedBehavior, "Parameters": sp_behavior_param}
    """Speed behavior"""
    mpc_param: NMPCLatKinContPVParam = NMPCLatKinContPVParam(n_horizon=n_horizon, t_step=t_step,
                                                             state_mult=state_mult, input_mult=input_mult,
                                                             delta_input_mult=delta_input_mult, technique=technique)
    mpc_controller = {"Name": "MPC Controller", "Controller": NMPCLatKinContPV, "Parameters": mpc_param}
    """MPC Controller"""
    steering_param: SCIdentityParam = SCIdentityParam()
    steering_controller = {"Name": "Identity controller", "Controller": SCIdentity, "Parameters": steering_param}
    """Pure Pursuit Controller"""
    metrics = [DeviationLateral, DeviationVelocity]
    """Metrics"""

    test_pp = TestController(scenario_name, "-", metrics, mpc_controller, sp_behavior, steering_controller, sp_controller)
    test_pp.run()
    test_pp.evaluate_metrics()
    test_pp.evaluate_metrics_test()
    test_pp.to_json()


test_mpckin_path_var()
