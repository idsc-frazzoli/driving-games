from sim_tests.controllers_tests.test_controller import TestController, Metric
from dg_commons.controllers.speed import SpeedBehavior, SpeedController, SpeedControllerParam, SpeedBehaviorParam
from dg_commons.controllers.stanley_controller import StanleyParam, Stanley
from dg_commons.controllers.steering_controllers import SCP, SCPParam
from dg_commons.analysis.metrics import DeviationLateral, DeviationVelocity
from typing import List


def test_stanley():
    scenario_name: str = "USA_Peach-1_1_T-1"
    # scenario_name: str = "ZAM_Tjunction-1_129_T-1"
    # scenario_name: str = "ARG_Carcarana-1_1_T-1"
    """Name of the chosen scenario"""
    vehicle_speed: float = 8
    """Nominal speed of the vehicle"""
    stanley_gain: float = 2
    """Scaling constant for speed dependent params"""
    ddelta_kp: float = 10
    """Proportional gain ddelta with respect to delta error"""
    speed_kp: float = 1
    """Propotioanl gain longitudinal speed controller"""
    speed_ki: float = 0.01
    """Integral gain longitudinal speed controller"""
    speed_kd: float = 0.1
    """Derivative gain longitudinal speed controller"""

    sp_controller_param: SpeedControllerParam = SpeedControllerParam(kP=speed_kp, kI=speed_ki, kD=speed_kd)
    sp_controller = {"Name": "Speed Controller", "Controller": SpeedController, "Parameters": sp_controller_param}
    """Speed Controller"""
    sp_behavior_param: SpeedBehaviorParam = SpeedBehaviorParam(nominal_speed=vehicle_speed)
    sp_behavior = {"Name": "Speed Behavior", "Behavior": SpeedBehavior, "Parameters": sp_behavior_param}
    """Speed behavior"""
    stanley_param: StanleyParam = StanleyParam(gain=stanley_gain)
    stanley_controller = {"Name": "Stanley Controller", "Controller": Stanley, "Parameters": stanley_param}
    """Stanley Controller"""
    steering_param: SCPParam = SCPParam(ddelta_kp=ddelta_kp)
    steering_controller = {"Name": "P controller", "Controller": SCP, "Parameters": steering_param}
    """Pure Pursuit Controller"""
    metrics = [DeviationLateral, DeviationVelocity]
    """Metrics"""

    test_pp = TestController(scenario_name, "-", metrics, stanley_controller, sp_behavior, steering_controller, sp_controller)
    test_pp.run()
    test_pp.evaluate_metrics()
    test_pp.evaluate_metrics_test()
    test_pp.to_json()


test_stanley()