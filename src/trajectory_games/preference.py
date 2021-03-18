from queue import PriorityQueue
from typing import Type, Dict, Mapping, Set, NewType, Tuple
from decimal import Decimal as D

import os

from frozendict import frozendict
from networkx import DiGraph, is_directed_acyclic_graph, all_simple_paths, has_path
from yaml import safe_load

from preferences import (
    Preference,
    ComparisonOutcome,
    SmallerPreferredTol,
    INDIFFERENT,
    INCOMPARABLE,
    FIRST_PREFERRED,
    SECOND_PREFERRED,
)
from .config import config_dir
from .metrics_def import Metric, PlayerOutcome
from .metrics import get_metrics_set

__all__ = [
    "WeightedPreference",
    "PosetalPreference",
]


class WeightedPreference(Preference[PlayerOutcome]):
    """Compare the total weighted values between evaluated metrics"""

    weights: Mapping[Metric, D]

    _pref: SmallerPreferredTol = SmallerPreferredTol(D("5e-3"))
    _config: Mapping = None
    _metric_dict: Mapping[str, Metric] = None

    def __init__(self, weights_str: str):
        if WeightedPreference._config is None:
            filename = os.path.join(config_dir, "pref_nodes.yaml")
            with open(filename) as load_file:
                WeightedPreference._config = safe_load(load_file)
            WeightedPreference._metric_dict = {type(m).__name__: m for m in get_metrics_set()}

        self.weights = {self._metric_dict[k]: D(v) for k, v in self._config[weights_str].items()}

    @staticmethod
    def get_type() -> Type[PlayerOutcome]:
        return PlayerOutcome

    def evaluate(self, outcome: PlayerOutcome) -> D:
        w = D("0")
        for metric, weight in self.weights.items():
            w += D(outcome[metric].total) * weight
        return w

    def compare(self, a: PlayerOutcome, b: PlayerOutcome) -> ComparisonOutcome:
        return self._pref.compare(self.evaluate(a), self.evaluate(b))

    def __repr__(self):
        ret: str = ""
        for metric, weight in self.weights.items():
            if len(ret) > 0:
                ret += "\n"
            ret += f"{round(float(weight), 2)}*{type(metric).__name__}"
        return ret

    def __lt__(self, other: "WeightedPreference") -> bool:
        return len(self.weights) < len(other.weights)


metric_type = NewType("metric", WeightedPreference)


class PosetalPreference(Preference[PlayerOutcome]):
    _config: Mapping = None
    _complement = {FIRST_PREFERRED: SECOND_PREFERRED, SECOND_PREFERRED: FIRST_PREFERRED,
                   INDIFFERENT: INDIFFERENT, INCOMPARABLE: INCOMPARABLE}
    _cache: Dict[Tuple[PlayerOutcome, PlayerOutcome], ComparisonOutcome]
    graph: DiGraph
    node_dict: Dict[str, metric_type] = {}
    level_nodes: Mapping[int, Set[metric_type]]

    def __init__(self, pref_str: str, use_cache: bool = False):
        if PosetalPreference._config is None:
            filename = os.path.join(config_dir, "player_pref.yaml")
            with open(filename) as load_file:
                PosetalPreference._config = safe_load(load_file)

        # Build graph
        self.build_graph(pref_str)
        # Pre-processing to speed up outcome comparisons
        self.calculate_levels()
        self.use_cache = use_cache
        self._cache = {}

    def add_node(self, name: str) -> metric_type:
        if name not in self.node_dict:
            node = WeightedPreference(weights_str=name)
            self.graph.add_node(node)
            self.node_dict[name] = node
        return self.node_dict[name]

    def build_graph(self, pref_str: str):
        self.graph = DiGraph()
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
                new_deg = (len(max(all_simple_paths(self.graph, source=root, target=node),
                                   key=lambda x: len(x))) - 1)
                level = max(level, new_deg)
            if level not in level_nodes:
                level_nodes[level] = set()
            level_nodes[level].add(node)
            self.graph.nodes[node]["level"] = level

        # Grid layout for visualisation
        scale = 40.0
        for deg, nodes in level_nodes.items():
            n_nodes = len(nodes)
            start = -(n_nodes - 1) / 2 - (0.0 if n_nodes % 2 == 1 else 0.2)
            i = 0
            for node in nodes:
                self.graph.nodes[node]["x"] = (start + i) * scale * 2.0
                self.graph.nodes[node]["y"] = -deg * scale * 0.4
                i = i + 1
        self.level_nodes = level_nodes

    @staticmethod
    def get_type() -> Type[PlayerOutcome]:
        return PlayerOutcome

    def compare(self, a: PlayerOutcome, b: PlayerOutcome) -> ComparisonOutcome:

        if self.use_cache:
            if isinstance(a, dict): a = frozendict(a)
            if isinstance(b, dict): b = frozendict(a)
            if (a, b) in self._cache:
                return self._cache[(a, b)]
            if (b, a) in self._cache:
                return self._complement[(self._cache[(b, a)])]
        OPEN = PriorityQueue(self.graph.number_of_nodes())
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
