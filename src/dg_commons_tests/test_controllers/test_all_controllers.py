from dg_commons_tests.test_controllers.controllers_to_test import *
from dg_commons_tests.test_controllers.controller_scenarios.scenario_to_test import scenarios
from dg_commons_tests.test_controllers.controller_test_utils import Select, TestInstance
from dg_commons.analysis.metrics import *


state_estimator = ExtendedKalman
state_estimator_params = ExtendedKalmanParam(
    actual_model_var=SemiDef([i*1 for i in [0.0001, 0.0001, 0.0001, 0.0001, 0.0001]]),
    actual_meas_var=SemiDef([i*0 for i in [0.001, 0.001, 0.001, 0.001, 0.001]]),

    belief_model_var=SemiDef([i * 1 for i in [0.0001, 0.0001, 0.0001, 0.0001, 0.0001]]),
    belief_meas_var=SemiDef([i * 0 for i in [0.001, 0.001, 0.001, 0.001, 0.001]]),

    initial_variance=SemiDef(matrix=np.zeros((5, 5))),

    dropping_technique=LGB,
    dropping_params=LGBParam(
        failure_p=0.0
    )
)


scenarios_to_test = [Select(scenarios["lane_change_left"], False),
                     Select(scenarios["turn_90_right"], True),
                     Select(scenarios["turn_90_left"], False),
                     Select(scenarios["small_snake"], False),
                     Select(scenarios["u-turn"], False),
                     Select(scenarios["left_cont_curve"], False),
                     Select(scenarios["race"], False)]

controllers_to_test = [
    Select(TestLQR, False),
    Select(TestPurePursuit, False),
    Select(TestStanley, False),
    Select(TestNMPCFullKinContPV, True),
    Select(TestNMPCFullKinDisPV, True),
    Select(TestNMPCFullKinContAN, True),
    Select(TestNMPCLatKinContPV, True),
    Select(TestNMPCLatKinDisPV, True),
    Select(TestNMPCLatKinContAN, True)
]

metrics_to_test = [
                Select(DeviationLateral, True),
                Select(DeviationVelocity, True),
                Select(SteeringVelocity, True),
                Select(Acceleration, True)
]

assert set(scenarios.keys()) == set([item.item.fig_name for item in scenarios_to_test])
helper = [item.item for item in controllers_to_test]
assert all([contr in helper for contr in contr_to_test])
assert all([contr in contr_to_test for contr in helper])
helper = [item.item for item in metrics_to_test]
assert all([metr in helper for metr in metrics_list])
assert all([metr in metrics_list for metr in helper])


metrics_to_test = [met.item for met in metrics_to_test if met.test]
for controllers_to_test in controllers_to_test:
    if controllers_to_test.test:
        controllers_to_test.item.state_estimator = state_estimator
        controllers_to_test.item.state_estimator_params = state_estimator_params
        for scenario_to_test in scenarios_to_test:
            if scenario_to_test.test:
                scenario = scenario_to_test.item
                test = TestInstance(controllers_to_test.item, metric=metrics_to_test, scenario=scenario_to_test.item)
                test.run()
