from queue import PriorityQueue
from typing import Type, Dict, Mapping, Set
from decimal import Decimal as D

import os
from networkx import DiGraph, read_adjlist, is_directed_acyclic_graph, all_simple_paths, has_path

from preferences import Preference, ComparisonOutcome, SmallerPreferredTol, \
    INDIFFERENT, INCOMPARABLE, FIRST_PREFERRED, SECOND_PREFERRED
from .metrics_def import Metric, EvaluatedMetric, PlayerOutcome

__all__ = [
    "EvaluatedMetricPreference",
    "PosetalPreference",
]


class EvaluatedMetricPreference(Preference[EvaluatedMetric]):
    """Compare the total values between evaluated metrics, doesn't check for types"""

    pref: SmallerPreferredTol = SmallerPreferredTol(D("1e-6"))

    @staticmethod
    def get_type() -> Type[EvaluatedMetric]:
        return EvaluatedMetric

    @staticmethod
    def compare(a: EvaluatedMetric, b: EvaluatedMetric) -> ComparisonOutcome:
        return EvaluatedMetricPreference.pref.compare(a.total, b.total)


class PosetalPreference(Preference[PlayerOutcome]):
    graph: DiGraph
    metric_dict: Mapping[str, Metric]
    level_nodes: Mapping[int, Set[str]]

    def __init__(self, pref_file: str, keys: Set[Metric]):

        self.build_graph_from_file(pref_file)

        # Create dict from metric names to metrics
        metric_dict: Dict[str, Metric] = {}
        for metric in keys:
            metric_dict[type(metric).__name__] = metric
        for node in self.graph:
            assert node in metric_dict, f"{node} not found in {metric_dict.keys()}"
        self.metric_dict = metric_dict

        # Pre-processing to speed up outcome comparisons
        self.calculate_levels()

    def build_graph_from_file(self, pref_file):

        # Ensure file exists
        assert os.path.isfile(pref_file), f"{pref_file} does not exist!"

        # Parsing doesn't like empty lines, so clean them up
        with open(pref_file) as filehandle:
            lines = filehandle.readlines()
        with open('graph', 'w') as filehandle:
            lines = filter(lambda x: x.strip(), lines)
            filehandle.writelines(lines)

        # Create graph from file
        self.graph = read_adjlist('graph', create_using=DiGraph(), nodetype=str)
        assert is_directed_acyclic_graph(self.graph)

        # Clean up
        os.remove('graph')

    def calculate_levels(self):
        level_nodes: Dict[int, Set[str]] = {}

        # Roots don't have input edges, degree = 0
        roots = [n for n, d in self.graph.in_degree() if d == 0]
        level_nodes[0] = set(roots)
        for root in roots:
            self.graph.nodes[root]['level'] = 0

        # Find longest path to edge from any root - assign as degree
        for node in self.graph.nodes:
            if node in roots: continue
            degree = 0
            for root in roots:
                new_deg = len(max(all_simple_paths(self.graph, source=root, target=node), key=lambda x: len(x))) - 1
                degree = max(degree, new_deg)
            if degree not in level_nodes: level_nodes[degree] = set()
            level_nodes[degree].add(node)
            self.graph.nodes[node]['level'] = degree

        # Grid layout for visualisation
        scale = 25.
        for deg, nodes in level_nodes.items():
            n_nodes = len(nodes)
            start = -(n_nodes-1)/2 - (.0 if n_nodes % 2 == 1 else .2)
            i = 0
            for node in nodes:
                self.graph.nodes[node]['x'] = (start + i)*scale*2.0
                self.graph.nodes[node]['y'] = -deg*scale*0.4
                i = i+1
        self.level_nodes = level_nodes

    def get_type(self) -> Type[PlayerOutcome]:
        return PlayerOutcome

    def compare(self, a: PlayerOutcome, b: PlayerOutcome) -> ComparisonOutcome:

        OPEN = PriorityQueue(self.graph.number_of_nodes())
        DONE: Set[str] = set()
        CLOSED: Set[str] = set()
        OUTCOMES: Set[ComparisonOutcome] = set()

        for root in self.level_nodes[0]:
            OPEN.put((0, root))

        while OPEN.qsize() > 0:
            if INCOMPARABLE in OUTCOMES or {FIRST_PREFERRED, SECOND_PREFERRED} <= OUTCOMES:
                return INCOMPARABLE
            _, metric = OPEN.get()
            if metric in DONE: continue
            DONE.add(metric)
            connected = False
            for closed in CLOSED:
                if has_path(G=self.graph, source=closed, target=metric): connected = True
            if connected: continue
            metric_type: Metric = self.metric_dict[metric]
            outcome = EvaluatedMetricPreference.compare(a[metric_type], b[metric_type])
            if outcome == INDIFFERENT:
                for child in self.graph.successors(metric):
                    OPEN.put((self.graph.nodes[child]['level'], child))
            else:
                OUTCOMES.add(outcome)
                CLOSED.add(metric)

        if INCOMPARABLE in OUTCOMES or {FIRST_PREFERRED, SECOND_PREFERRED} <= OUTCOMES:
            return INCOMPARABLE
        if FIRST_PREFERRED in OUTCOMES:
            return FIRST_PREFERRED
        if SECOND_PREFERRED in OUTCOMES:
            return SECOND_PREFERRED

        return INDIFFERENT
