from dg_commons_tests.test_controllers.controllers_to_test import *
from dg_commons_tests.test_controllers.controller_test_utils import DT
from dg_commons_tests.test_controllers.controller_scenarios.scenario_to_test import scenarios
from dg_commons_tests.test_controllers.controller_test_utils import Select, TestInstance
from dg_commons.state_estimators.estimator_types import *
from dg_commons.analysis.metrics import *
import time
import matplotlib.pyplot as plt
import json


class Verbosity:
    def __init__(self, val: int):
        assert val in [0, 1, 2]
        self._val = val

    @property
    def val(self):
        return self._val


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

n_scenarios = sum([1 for s in scenarios_to_test if s.test])
n_controllers = sum([s.item.get_count() for s in controllers_to_test if s.test])
steps: int = n_controllers*n_scenarios

if verbosity.val > 0:
    print("[Testing Cases]... There are {} testing cases".format(steps))

counter = 0
times = []
controllers = []
scenarios = []
timing = {}

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
                    timing[dict_name] = {}
                    counter += 1
                    scenario = scenario_to_test.item
                    test = TestInstance(controller_to_test, metric=metrics_to_test, scenario=scenario_to_test.item)
                    test.run()
                    delta = test.avg_t
                    if verbosity.val > 0:
                        print("[Testing]...")
                        print("[Controller]...", controllers_to_test.item.folder_name)
                        print("[Scenario]...", scenario.fig_name)
                        print("[Percentage]...", round(counter/steps*100, 3), "%")
                        print("[Time]... took {} seconds".format(round(delta, 3)))

                    times.append(delta)
                    controllers.append(controllers_to_test.item.folder_name)
                    scenarios.append(scenario.fig_name)
                    timing[dict_name][scenario.fig_name] = delta
                    """ For post processing """
t2 = time.time()
if verbosity.val > 0:
    print("The whole process took {} seconds".format(round(t2 - t1, 3)))

name = "OVERALL STATISTICS"
output_dir = os.path.join("out", "simulation_timing_statistics")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def create_histograms(ch_name, ch_times, ch_controllers):
    if verbosity.val > 1:
        print(ch_name)
        print("Average: {} seconds".format(round(sum(ch_times) / len(ch_times), 3)))
        maximum = max(ch_times)
        idx = ch_times.index(maximum)
        scene = ", Scenario: " + scenarios[idx] if name == "OVERALL STATISTICS" else ""
        print("Max is {} for: ".format(round(maximum, 3)),
              " Controller: {}{}".format(ch_controllers[idx], scene))
        minimum = min(ch_times)
        idx = ch_times.index(minimum)
        scene = ", Scenario: " + scenarios[idx] if name == "OVERALL STATISTICS" else ""
        print("Min is {} for: ".format(round(minimum, 3)),
              " Controller: {}{}".format(ch_controllers[idx], scene))
        print()
    plt.hist(ch_times)
    plt.title(ch_name)
    plt.xlabel("Time [s]")
    plt.savefig(os.path.join(output_dir, ch_name))
    plt.clf()


scenario_list = list(dict.fromkeys(scenarios))
controller_list = list(dict.fromkeys(controllers))
integer_values = [i+1 for i in range(len(scenario_list))]
mapping = dict(zip(scenario_list, integer_values))

create_histograms(name, times, controllers)

for scene in scenario_list:
    temp = [(times[i], controllers[i]) for i, t in enumerate(scenarios) if t == scene]
    scene_times = [i[0] for i in temp]
    scene_controllers = [i[1] for i in temp]
    name = "STATISTICS ABOUT " + scene.upper()
    create_histograms(name, scene_times, scene_controllers)

'''
def create_plots(ch_name, ch_times, ch_scenarios):
    scenario_values = [mapping[c] for c in ch_scenarios]
    plt.scatter(scenario_values, ch_times, label=ch_name)
    plt.scatter([integer_values[-1] + int(n_scenarios/2)], [0])
    plt.title("Timing")
    plt.ylabel("Time [s]")
    plt.xlabel("Scenarios")


for contr in controller_list:
    temp = [(times[i], scenarios[i]) for i, t in enumerate(controllers) if t == contr]
    contr_times = [i[0] for i in temp]
    contr_scenario = [i[1] for i in temp]
    create_plots(contr, contr_times, contr_scenario)
# plt.xticks(integer_values, scenario_list)
plt.xticks(integer_values)
plt.legend(bbox_to_anchor=(1.1, 1.05))
col_labels = ['val']
row_labels = scenario_list
table_vals = [integer_values]
# the rectangle is where I want to place the table
the_table = plt.table(cellText=table_vals,
                      colWidths=[0.1]*3,
                      rowLabels=row_labels,
                      colLabels=col_labels,
                      loc='right')
plt.text(12, 3.4, '', size=8)

plt.savefig(os.path.join(output_dir, "timing"))
'''

json_filename = os.path.join(output_dir, "timing_data.json")
json_file = open(json_filename, "w")
json.dump(timing, json_file)
json_file.close()
