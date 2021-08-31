from dg_commons.analysis.metrics import DeviationLateral
from sim_tests.controllers_tests.test_controller import TestController
from dg_commons.controllers.speed import SpeedBehavior, SpeedController, SpeedControllerParam, SpeedBehaviorParam
from dg_commons.controllers.pure_pursuit import PurePursuit, PurePursuitParam
from dg_commons.controllers.steering_controllers import SCP, SCPParam


def test_pure_pursuit():
    scenario_name: str = "USA_Peach-1_1_T-1"
    # scenario_name: str = "ZAM_Tjunction-1_129_T-1"
    # scenario_name: str = "ARG_Carcarana-1_1_T-1"
    """Name of the chosen scenario"""
    vehicle_speed: float = 5
    """Nominal speed of the vehicle"""
    k_lookahead: float = 1.8
    """Scaling constant for speed dependent params"""
    ddelta_kp: float = 10
    """Proportional gain ddelta with respect to delta error"""
    speed_kp: float = 2
    """Propotioanl gain longitudinal speed controller"""
    speed_ki: float = 0.005
    """Integral gain longitudinal speed controller"""
    speed_kd: float = 0.1
    """Derivative gain longitudinal speed controller"""

    sp_controller_param: SpeedControllerParam = SpeedControllerParam(kP=speed_kp, kI=speed_ki, kD=speed_kd)
    sp_controller = {"Name": "Speed Controller", "Controller": SpeedController, "Parameters": sp_controller_param}
    """Speed Controller"""
    sp_behavior_param: SpeedBehaviorParam = SpeedBehaviorParam(nominal_speed=vehicle_speed)
    sp_behavior = {"Name": "Speed Behavior", "Behavior": SpeedBehavior, "Parameters": sp_behavior_param}
    """Speed behavior"""
    pp_param: PurePursuitParam = PurePursuitParam(k_lookahead=k_lookahead)
    pp_controller = {"Name": "Pure Pursuit Controller", "Controller": PurePursuit, "Parameters": pp_param}
    """Pure Pursuit Controller"""
    steering_param: SCPParam = SCPParam(ddelta_kp=ddelta_kp)
    steering_controller = {"Name": "P controller", "Controller": SCP, "Parameters": steering_param}
    """Pure Pursuit Controller"""
    metrics = [DeviationLateral]
    """Metrics"""

    test_pp = TestController(scenario_name, "-", pp_controller, sp_controller, sp_behavior, steering_controller, metrics)
    test_pp.run()
    test_pp.evaluate_metrics()
    test_pp.evaluate_metrics_test()
    test_pp.to_json()


test_pure_pursuit()
