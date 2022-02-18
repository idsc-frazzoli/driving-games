import os
from typing import Mapping

from yaml import safe_load

from driving_games.metrics_structures import Metric
from trajectory_games import WeightedMetricPreference
from trajectory_games.config import CONFIG_DIR
from zuper_commons.types import ZValueError


class MetricNodesServer:
    name: str
    weights: Mapping[Metric, float]
    """ Weights of the different nodes. Each node can either be a metric or a weighted preference """

    _available: Mapping[str, WeightedMetricPreference]

    def __init__(self):
        # todo in the future option to load a specific config file
        filename = os.path.join(CONFIG_DIR, "pref_nodes.yaml")
        # core_metrics = get_metrics_set()
        with open(filename) as f:
            nodes = safe_load(f)
        print(nodes)
        # WeightedMetricPreference._metric_dict = {type(m).__name__: m for m in core_metrics}
        # todo init
        # self.name = weights_str
        # weights: Dict[AllMetrics, D] = {}
        for k, v in nodes.items():
            if k in self._available:
                raise ZValueError(f"Metric Node {k} is already defined", available=self._available)
            else:
                pass

        #         try:
        #             w_metric = WeightedMetricPreference(weights_str=k)
        #         except:
        #             raise ValueError(f"Key {k} not found in metrics or weighted metrics!")
        #         WeightedMetricPreference._metric_dict[k] = w_metric
        #     weights[WeightedMetricPreference._metric_dict[k]] = D(v)
        # self.weights = weights
