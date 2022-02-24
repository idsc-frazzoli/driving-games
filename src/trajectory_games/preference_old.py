import os
from decimal import Decimal as D
from queue import PriorityQueue
from typing import Dict, Mapping, NewType, Set, Tuple, Type, Union

from frozendict import frozendict
from networkx import all_simple_paths, DiGraph, has_path, is_directed_acyclic_graph
from yaml import safe_load

from driving_games.metrics_structures import Metric, PlayerEvaluatedMetrics
from preferences import (
    ComparisonOutcome,
    FIRST_PREFERRED,
    INCOMPARABLE,
    INDIFFERENT,
    Preference,
    SECOND_PREFERRED,
    SmallerPreferredTol,
)
from .config import CONFIG_DIR
from .config.ral import config_dir_ral
from .metrics import get_metrics_set

__all__ = [
    "WeightedMetricPreference",
    "PosetalPreference",
]

AllMetrics = Union[Metric, "WeightedPreference"]


class WeightedMetricPreference(Preference[PlayerEvaluatedMetrics]):
    """Compare the total weighted values between evaluated metrics"""

    name: str
    weights: Mapping[Metric, float]
    """ Weights of the different nodes. Each node can either be a metric or a weighted preference """

    """ Internal parameters """
    _pref: SmallerPreferredTol = SmallerPreferredTol(D("5e-3"))
    _config: Mapping = None
    _metric_dict: Dict[str, AllMetrics] = None

    def __init__(
        self,
        weights_str: str,
    ):
        if WeightedMetricPreference._config is None:
            filename = os.path.join(CONFIG_DIR, "pref_nodes.yaml")
            with open(filename) as load_file:
                WeightedMetricPreference._config = safe_load(load_file)
            WeightedMetricPreference._metric_dict = {type(m).__name__: m for m in get_metrics_set()}

        self.name = weights_str
        weights: Dict[AllMetrics, D] = {}
        for k, v in WeightedMetricPreference._config[weights_str].items():
            if k not in WeightedMetricPreference._metric_dict:
                try:
                    w_metric = WeightedMetricPreference(weights_str=k)
                except:
                    raise ValueError(f"Key {k} not found in metrics or weighted metrics!")
                WeightedMetricPreference._metric_dict[k] = w_metric
            weights[WeightedMetricPreference._metric_dict[k]] = D(v)
        self.weights = weights

    def get_type(
        self,
    ) -> Type[PlayerEvaluatedMetrics]:
        return PlayerEvaluatedMetrics

    def compare(self, a: PlayerEvaluatedMetrics, b: PlayerEvaluatedMetrics) -> ComparisonOutcome:
        return self._pref.compare(self.evaluate(a), self.evaluate(b))

    def evaluate(self, outcome: PlayerEvaluatedMetrics) -> D:
        w = D("0")
        for metric, weight in self.weights.items():
            value = (
                metric.evaluate(outcome=outcome)
                if isinstance(metric, WeightedMetricPreference)
                else D(outcome[metric].value)
            )
            w += value * weight
        return w

    def __repr__(self):
        ret: str = ""
        for metric, weight in self.weights.items():
            if len(ret) > 0:
                ret += "\n"
            met_str = metric.name if isinstance(metric, WeightedMetricPreference) else type(metric).__name__
            ret += f"{round(float(weight), 2)}*{met_str}"
        return ret

    def __lt__(self, other: "WeightedMetricPreference") -> bool:
        return len(self.weights) < len(other.weights)


metric_type = NewType("metric", WeightedMetricPreference)  # fixme this does not seem right


class PosetalPreference(Preference[PlayerEvaluatedMetrics]):
    """A preference specified over the various nodes.
    Each node is a metric or a weighted combination of metrics (or weighted nodes)"""

    graph: DiGraph
    """ Preference graph """
    level_nodes: Mapping[int, Set[metric_type]]
    """ All nodes used, and sorted by level """
    pref_str: str
    """ Name of preference """
    no_pref: bool = False
    """ No preference over all the outcomes? """

    # Internal parameters
    _config: Mapping = None
    _complement = {
        FIRST_PREFERRED: SECOND_PREFERRED,
        SECOND_PREFERRED: FIRST_PREFERRED,
        INDIFFERENT: INDIFFERENT,
        INCOMPARABLE: INCOMPARABLE,
    }
    _cache: Dict[Tuple[PlayerEvaluatedMetrics, PlayerEvaluatedMetrics], ComparisonOutcome]
    _node_dict: Dict[str, metric_type] = {}

    def __init__(self, pref_str: str, use_cache: bool = False):
        if PosetalPreference._config is None:
            filename = os.path.join(config_dir_ral, "player_pref.yaml")
            with open(filename) as load_file:
                PosetalPreference._config = safe_load(load_file)

        # Build graph
        self.build_graph(pref_str)
        # Pre-processing to speed up outcome comparisons
        self.calculate_levels()
        self.use_cache = use_cache
        self._cache = {}
        self.pref_str = pref_str

    def __eq__(self, other: "PosetalPreference"):
        if not isinstance(other, PosetalPreference):
            return False
        return self.pref_str == other.pref_str

    def __hash__(self):
        return hash(self.pref_str)

    def add_node(self, name: str) -> metric_type:
        if name not in PosetalPreference._node_dict:
            node = WeightedMetricPreference(weights_str=name)
            PosetalPreference._node_dict[name] = node
        else:
            node = PosetalPreference._node_dict[name]
        self.graph.add_node(node)
        return node

    def build_graph(self, pref_str: str):
        self.graph = DiGraph()
        if pref_str == "NoPreference":
            self.no_pref = True
            return
        if pref_str not in self._config:
            raise ValueError(f"{pref_str} not found in keys = {self._config.keys()}")
        for key, parents in self._config[pref_str].items():
            node = self.add_node(name=key)
            for p in parents:
                p_node = self.add_node(name=p)
                self.graph.add_edge(p_node, node)
        assert is_directed_acyclic_graph(self.graph)

    def calculate_levels(self):
        level_nodes: Dict[int, Set[metric_type]] = {}

        # Roots don't have input edges, degree = 0
        roots = [n for n, d in self.graph.in_degree() if d == 0]
        level_nodes[0] = set(roots)
        for root in roots:
            self.graph.nodes[root]["level"] = 0

        # Find longest path to edge from any root - assign as degree
        for node in self.graph.nodes:
            if node in roots:
                continue
            level = 0
            for root in roots:
                all_lens = [len(x) for x in all_simple_paths(self.graph, source=root, target=node)]
                if len(all_lens) > 0:
                    level = max(level, max(all_lens) - 1)
            if level not in level_nodes:
                level_nodes[level] = set()
            level_nodes[level].add(node)
            self.graph.nodes[node]["level"] = level

        # Grid layout for visualisation
        scale = 40.0
        for deg, nodes in level_nodes.items():
            n_nodes = len(nodes)
            start = -(n_nodes - 1) / 2
            i = 0
            for node in nodes:
                self.graph.nodes[node]["x"] = (start + i) * scale * 2.0
                self.graph.nodes[node]["y"] = -deg * scale * 0.4
                i = i + 1
        self.level_nodes = level_nodes

    def get_type(
        self,
    ) -> Type[PlayerEvaluatedMetrics]:
        return PlayerEvaluatedMetrics

    def compare(self, a: PlayerEvaluatedMetrics, b: PlayerEvaluatedMetrics) -> ComparisonOutcome:

        if self.no_pref:
            return INDIFFERENT
        if self.use_cache:
            if isinstance(a, dict):
                a = frozendict(a)
            if isinstance(b, dict):
                b = frozendict(a)
            if (a, b) in self._cache:
                return self._cache[(a, b)]
            if (b, a) in self._cache:
                return self._complement[(self._cache[(b, a)])]
        OPEN = PriorityQueue(100)
        DONE: Set[metric_type] = set()
        CLOSED: Set[metric_type] = set()
        OUTCOMES: Set[ComparisonOutcome] = set()

        for root in self.level_nodes[0]:
            OPEN.put((0, root))

        while OPEN.qsize() > 0:
            if INCOMPARABLE in OUTCOMES or {FIRST_PREFERRED, SECOND_PREFERRED} <= OUTCOMES:
                break
            _, metric = OPEN.get()
            if metric in DONE:
                continue
            DONE.add(metric)
            connected = False
            for closed in CLOSED:
                if has_path(G=self.graph, source=closed, target=metric):
                    connected = True
            if connected:
                continue
            outcome = metric.compare(a, b)
            if outcome == INDIFFERENT:
                for child in self.graph.successors(metric):
                    OPEN.put((self.graph.nodes[child]["level"], child))
            else:
                OUTCOMES.add(outcome)
                CLOSED.add(metric)

        ret: ComparisonOutcome = INDIFFERENT
        if INCOMPARABLE in OUTCOMES or {FIRST_PREFERRED, SECOND_PREFERRED} <= OUTCOMES:
            ret = INCOMPARABLE
        elif FIRST_PREFERRED in OUTCOMES:
            ret = FIRST_PREFERRED
        elif SECOND_PREFERRED in OUTCOMES:
            ret = SECOND_PREFERRED

        if self.use_cache:
            self._cache[(a, b)] = ret
        return ret
