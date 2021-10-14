from dg_commons_dev_tests.test_controllers.controllers_to_test import *
from dg_commons_dev_tests.test_controllers.controller_test_utils import Select, Verbosity, \
    TestMultipleControllerInstances
from dg_commons_dev.analysis.metrics import *
from dg_commons_dev_tests.test_controllers.controller_scenarios.scenario_to_test import scenarios

if __name__ == '__main__':
    verbosity: Verbosity = Verbosity(2)

    scenarios_to_test = [Select(scenarios["lane_change_left"], True),
                         Select(scenarios["turn_90_right"], True),
                         Select(scenarios["turn_90_left"], True),
                         Select(scenarios["small_snake"], True),
                         Select(scenarios["u-turn"], True),
                         Select(scenarios["left_cont_curve"], True),
                         Select(scenarios["vertical"], True),
                         Select(scenarios["race"], False)]

    controllers_to_test = [
        Select(TestLQR, True),
        Select(TestPurePursuit, True),
        Select(TestStanley, True),
        Select(TestNMPCFullKinContPV, True), Select(TestNMPCFullKinContAN, True),
        Select(TestNMPCFullKinDisPV, True), Select(TestNMPCFullKinDisAN, True),
        Select(TestNMPCLatKinContPV, True), Select(TestNMPCLatKinContAN, True),
        Select(TestNMPCLatKinDisPV, True), Select(TestNMPCLatKinDisAN, True)
    ]

    metrics_to_test = [
                    Select(DeviationLateral, True),
                    Select(DeviationVelocity, True),
                    Select(SteeringVelocity, True),
                    Select(Acceleration, True),
                    Select(DTForCommand, True)
    ]

    Test = TestMultipleControllerInstances(controllers_to_test, metrics_to_test, scenarios_to_test, verbosity)
    Test.run()
