from driving_games.metrics_structures import MetricNodeName
from trajectory_games import MetricNodePreference


def test_weighted_metric_node_preference():
    weightsd_4 = MetricNodeName("weights_4")
    test = MetricNodePreference(weightsd_4)
    print(test)

    weights_ral_1 = MetricNodeName("weights_ral_1")
    test_2 = MetricNodePreference(weights_ral_1)
    print(test_2)
