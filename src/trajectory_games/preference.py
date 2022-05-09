import os
from decimal import Decimal as D
from heapq import *
from typing import Type, Dict, Mapping, Set, Tuple

from networkx import DiGraph, is_directed_acyclic_graph, all_simple_paths, has_path
from yaml import safe_load

from dg_commons import fd
from dg_commons.time import time_function
from driving_games.metrics_structures import Metric, PlayerEvaluatedMetrics, MetricNodeName
from preferences import (
    Preference,
    ComparisonOutcome,
    SmallerPreferredTol,
    INDIFFERENT,
    INCOMPARABLE,
    FIRST_PREFERRED,
    SECOND_PREFERRED,
)
from .config import CONFIG_DIR
from .config.ral import config_dir_ral
from .metrics import *

__all__ = [
    "MetricNodePreference",
    "PosetalPreference",
]

# todo move the generation of WeightedMetirc to a separate "factory"?


class MetricNodePreference(Preference[PlayerEvaluatedMetrics]):
    """A MetricNodePreference is a Metric node of the posetal preference.
    As a metric, it is an aggregation (weighted sum) of basic metrics.
    Itself contains also the preference on the node itself (scalar smaller preferred for rules-like)."""

    name: MetricNodeName
    weights: Mapping[Metric, float]
    """ Weights of the different nodes. Each node can either be a metric or a weighted preference """

    """ Internal parameters """
    _pref: SmallerPreferredTol = SmallerPreferredTol(D(0))  # SmallerPreferredTol(D("5e-3"))
    _config: Mapping = None
    _metric_dict: Dict[MetricNodeName, Metric] = None

    def __init__(self, weights_str: MetricNodeName):
        if MetricNodePreference._config is None:
            filename = os.path.join(CONFIG_DIR, "pref_nodes.yaml")
            with open(filename) as load_file:
                MetricNodePreference._config = safe_load(load_file)
            MetricNodePreference._metric_dict = {type(m).__name__: m for m in get_metrics_set()}

        self.name = weights_str
        weights: Dict[Metric, float] = {}
        for k, v in MetricNodePreference._config[weights_str].items():
            if k not in MetricNodePreference._metric_dict:
                try:
                    w_metric = MetricNodePreference(weights_str=k)
                except:
                    raise ValueError(f"Key {k} not found in metrics or weighted metrics!")
                MetricNodePreference._metric_dict[k] = w_metric
            weights[MetricNodePreference._metric_dict[k]] = float(v)
        self.weights = fd(weights)

    def get_type(
        self,
    ) -> Type[PlayerEvaluatedMetrics]:
        # fixme
        return PlayerEvaluatedMetrics

    def compare(self, a: PlayerEvaluatedMetrics, b: PlayerEvaluatedMetrics) -> ComparisonOutcome:
        return self._pref.compare(self.evaluate(a), self.evaluate(b))

    def evaluate(self, outcome: PlayerEvaluatedMetrics) -> float:
        val = 0
        for metric, weight in self.weights.items():
            m_value = outcome[metric].value
            val += m_value * weight
        return val

    def __repr__(self):
        ret: str = ""
        for metric, weight in self.weights.items():
            if len(ret) > 0:
                ret += "\n"
            met_str = metric.name if isinstance(metric, MetricNodePreference) else type(metric).__name__
            ret += f"{round(float(weight), 2)}*{met_str}"
        return ret

    def __lt__(self, other: "MetricNodePreference") -> bool:
        # todo remove? not formally well defined
        return len(self.weights) < len(other.weights)

    def get_name(self) -> MetricNodeName:
        return self.name


class PosetalPreference(Preference[PlayerEvaluatedMetrics]):
    """A preference specified over several nodes.
    Each node is a metric or a weighted combination of metrics (or weighted nodes), see WeightedMetricPreference"""

    graph: DiGraph
    """ Preference graph """
    nodes_level: Mapping[int, Set[MetricNodePreference]]
    """ All nodes used, and sorted by level """
    pref_str: str
    """ Name of preference """
    no_pref: bool = False
    """ No preference over all the outcomes? """

    # Internal parameters
    _config: Mapping = None
    _complement = fd(
        {
            FIRST_PREFERRED: SECOND_PREFERRED,
            SECOND_PREFERRED: FIRST_PREFERRED,
            INDIFFERENT: INDIFFERENT,
            INCOMPARABLE: INCOMPARABLE,
        }
    )
    _cache: Dict[Tuple[PlayerEvaluatedMetrics, PlayerEvaluatedMetrics], ComparisonOutcome]
    _node_dict: Dict[str, MetricNodePreference] = {}

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

    def add_node(self, name: str) -> MetricNodePreference:
        if name not in PosetalPreference._node_dict:
            node = MetricNodePreference(weights_str=MetricNodeName(name))
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
        level_nodes: Dict[int, Set[MetricNodePreference]] = {}

        # Roots don't have input edges, degree = 0
        roots = [n for n, d in self.graph.in_degree() if d == 0]
        level_nodes[0] = set(roots)
        for root in roots:
            self.graph.nodes[root]["level"] = 0

        # Find the longest path to edge from any root - assign as degree
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
                self.graph.nodes[node]["x"] = (start + i) * scale * 1.0
                self.graph.nodes[node]["y"] = -deg * scale * 1.0
                i = i + 1
        self.nodes_level = level_nodes

    def get_type(
        self,
    ) -> Type[PlayerEvaluatedMetrics]:
        return PlayerEvaluatedMetrics

    @time_function
    def compare(self, a: PlayerEvaluatedMetrics, b: PlayerEvaluatedMetrics) -> ComparisonOutcome:
        """
        This function compares outcomes through a preordered preference.
        :param a: first set of evaluated metrics (first trajectory)
        :param b: second set of evaluated metrics (second trajectory)
        :return: outcome, i.e. one of: first preferred, second preferred, indifferent or incomparable
        """
        if self.no_pref:
            return INDIFFERENT

        #  initialize queue with top level nodes of preference graph
        queue = []
        for root in self.nodes_level[0]:
            heappush(queue, (0, root))

        # list containing results of node comparisons
        outcomes = []
        # list of metrics that are not equal between a and b that allow discriminating between those two outcomes
        discriminants = []

        while len(queue) > 0:
            # check if the outcomes are incomparable
            if INCOMPARABLE in outcomes or (FIRST_PREFERRED in outcomes and SECOND_PREFERRED in outcomes):
                return INCOMPARABLE
            # extract next element from queue
            _, metric = heappop(queue)
            # compute comparison for a single metric (e.g. smaller preferred)
            comp = metric.compare(a, b)

            # if one metric is incomparable, return incomparable for the entire preference structure
            if comp == INCOMPARABLE:
                return INCOMPARABLE

            skip = False
            # skip all children when the parent node already yields a comparison that is not indifferent
            for discr in discriminants:
                if has_path(G=self.graph, source=discr, target=metric):
                    skip = True
            if skip:
                continue

            # if first or second outcome are preferred, store result and metric
            if comp == FIRST_PREFERRED or comp == SECOND_PREFERRED:
                outcomes.append(comp)
                discriminants.append(metric)

            # if a node yields indifferent, add its successors to queue
            if comp == INDIFFERENT:
                for child in self.graph.successors(metric):
                    heappush(queue, (self.graph.nodes[child]["level"], child))

        # final step: combine outcomes
        if FIRST_PREFERRED in outcomes:
            return FIRST_PREFERRED
        elif SECOND_PREFERRED in outcomes:
            return SECOND_PREFERRED

        # sanity check
        assert (
            FIRST_PREFERRED not in outcomes and SECOND_PREFERRED not in outcomes
        ), "Something went wrong in condition checking"

        return INDIFFERENT

    # @time_function
    # def compare_old(self, a: PlayerEvaluatedMetrics, b: PlayerEvaluatedMetrics) -> ComparisonOutcome:
    #
    #     if self.no_pref:
    #         return INDIFFERENT
    #     if self.use_cache:
    #         if isinstance(a, dict):
    #             a = frozendict(a)
    #         if isinstance(b, dict):
    #             b = frozendict(a)
    #         if (a, b) in self._cache:
    #             return self._cache[(a, b)]
    #         if (b, a) in self._cache:
    #             return self._complement[(self._cache[(b, a)])]
    #     OPEN = PriorityQueue(100)
    #     DONE: Set[WeightedMetricPreference] = set()
    #     CLOSED: Set[WeightedMetricPreference] = set()
    #     OUTCOMES: Set[ComparisonOutcome] = set()
    #
    #     for root in self.level_nodes[0]:
    #         OPEN.put((0, root))
    #
    #     while OPEN.qsize() > 0:
    #         if INCOMPARABLE in OUTCOMES or {FIRST_PREFERRED, SECOND_PREFERRED} <= OUTCOMES:
    #             break
    #         _, metric = OPEN.get()
    #         if metric in DONE:
    #             continue
    #         DONE.add(metric)
    #         connected = False
    #         for closed in CLOSED:
    #             if has_path(G=self.graph, source=closed, target=metric):
    #                 connected = True
    #         if connected:
    #             continue
    #         outcome = metric.compare(a, b)
    #         if outcome == INDIFFERENT:
    #             for child in self.graph.successors(metric):
    #                 OPEN.put((self.graph.nodes[child]["level"], child))
    #         else:
    #             OUTCOMES.add(outcome)
    #             CLOSED.add(metric)
    #
    #     ret: ComparisonOutcome = INDIFFERENT
    #     if INCOMPARABLE in OUTCOMES or {FIRST_PREFERRED, SECOND_PREFERRED} <= OUTCOMES:
    #         ret = INCOMPARABLE
    #     elif FIRST_PREFERRED in OUTCOMES:
    #         ret = FIRST_PREFERRED
    #     elif SECOND_PREFERRED in OUTCOMES:
    #         ret = SECOND_PREFERRED
    #
    #     if self.use_cache:
    #         self._cache[(a, b)] = ret
    #     return ret
