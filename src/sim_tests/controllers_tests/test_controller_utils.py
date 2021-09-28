from dg_commons.controllers.full_controller_base import VehicleController
from sim_tests.controllers_tests.test_controller import TestController
from dg_commons.controllers.speed import SpeedBehavior


def run_test(controller: VehicleController,  scenario_name: str):

    sp_behavior = {"Name": "Speed Behavior",
                   "Behavior": SpeedBehavior,
                   "Parameters": controller.speed_behavior_param}

    main_controller = {"Name": controller.controller.__name__,
                       "Controller": controller.controller,
                       "Parameters": controller.controller_params}

    steering_controller = {"Name": controller.steering_controller.__class__.__name__,
                           "Controller": controller.steering_controller,
                           "Parameters": controller.steering_controller_params}

    longitudinal_controller = {"Name": controller.longitudinal_controller.__class__.__name__,
                               "Controller": controller.longitudinal_controller,
                               "Parameters": controller.longitudinal_controller_params} \
        if controller.longitudinal_controller is not None else None

    state_estimator = {"Name": controller.state_estimator.__class__.__name__,
                       "Estimator": controller.state_estimator,
                       "Parameters": controller.state_estimator_params} \
        if controller.state_estimator is not None else None

    test_pp = TestController(scenario_name, "-", controller.metrics, controller.lf_agent, main_controller,
                             sp_behavior, steering_controller, longitudinal_controller, state_estimator)
    test_pp.run()
    test_pp.evaluate_metrics()
    # test_pp.to_json()
