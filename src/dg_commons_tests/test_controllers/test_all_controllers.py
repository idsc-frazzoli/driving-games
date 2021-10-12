import numpy as np
from dg_commons_tests.test_controllers.controllers_to_test import *
from dg_commons_tests.test_controllers.controller_test_utils import Select, TestInstance, Verbosity, DataCollect
from dg_commons.analysis.metrics import *
import time
import matplotlib.pyplot as plt
from dg_commons_tests.test_controllers.controller_scenarios.scenario_to_test import scenarios
import json


def simulate_all_test_instances(scenarios_to_test, controllers_to_test,
                                metrics_to_test, verbosity: Verbosity = Verbosity(1), scenarios=scenarios):

    assert set(scenarios.keys()) == set([item.item.fig_name for item in scenarios_to_test])
    helper = [item.item for item in controllers_to_test]
    assert all([contr in helper for contr in contr_to_test])
    assert all([contr in contr_to_test for contr in helper])
    helper = [item.item for item in metrics_to_test]
    assert all([metr in helper for metr in metrics_list])
    assert all([metr in metrics_list for metr in helper])
    metrics_to_test = [met.item for met in metrics_to_test if met.test]

    n_scenarios = sum([1 for s in scenarios_to_test if s.test])
    n_controllers = sum([s.item.get_count() for s in controllers_to_test if s.test])
    steps: int = n_controllers * n_scenarios

    if verbosity.val > 0:
        print("[Testing Cases]... There are {} testing cases".format(steps))

    counter = 0
    keys = {key.__name__ for key in metrics_to_test}
    data = DataCollect(keys)

    t1 = time.time()
    for controllers_to_test in controllers_to_test:
        if controllers_to_test.test:
            it = 0
            for controller_to_test in controllers_to_test.item.gen():
                controller_to_test.add_sub_folder("Test{}".format(it))
                root_name = controller_to_test.folder_name + "Test{}".format(it)
                it += 1
                for scenario_to_test in scenarios_to_test:
                    if scenario_to_test.test:
                        dict_name = root_name + scenario_to_test.item.fig_name
                        counter += 1
                        scenario = scenario_to_test.item
                        test = TestInstance(controller_to_test, metric=metrics_to_test, scenario=scenario_to_test.item)
                        result = test.run()
                        data.collect_data(dict_name, result, scenario.fig_name, round(counter / steps * 100, 3),
                                          controllers_to_test.item.folder_name, verbosity)

    t2 = time.time()
    if verbosity.val > 0:
        print("The whole process took {} seconds".format(round(t2 - t1, 3)))

    data.show_data(verbosity)


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

    simulate_all_test_instances(scenarios_to_test, controllers_to_test, metrics_to_test, verbosity)
