from queue import PriorityQueue
from typing import Type, Set

from dg_commons import fd
from networkx import DiGraph

from driving_games.metrics_structures import PlayerEvaluatedMetrics
from preferences import (
    Preference,
    ComparisonOutcome,
    INCOMPARABLE,
    FIRST_PREFERRED,
    SECOND_PREFERRED,
    INDIFFERENT,
    SmallerPreferred,
)


class PosetalPrefBuilder:
    # todo bunch of utility methods that return a PosetalPref
    pass


class PosetalPref(Preference[PlayerEvaluatedMetrics]):
    """A preference specified over the various nodes.
    Each node is a metric or a weighted combination of metrics (or weighted nodes)"""

    G: DiGraph
    """ The graph representing the priority over the nodes"""
    # fixme this graph should have a level
    metric_pref: SmallerPreferred

    def __hash__(self):
        return hash(self.pref_str)

    def get_type(self) -> Type[PlayerEvaluatedMetrics]:
        # todo
        return PlayerEvaluatedMetrics

    def compare(self, a: PlayerEvaluatedMetrics, b: PlayerEvaluatedMetrics) -> ComparisonOutcome:

        if self.no_pref:
            return INDIFFERENT
        if self.use_cache:
            if isinstance(a, dict):
                a = fd(a)
            if isinstance(b, dict):
                b = fd(a)
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
                if has_path(G=self.G, source=closed, target=metric):
                    connected = True
            if connected:
                continue
            outcome = metric.compare(a, b)
            if outcome == INDIFFERENT:
                for child in self.G.successors(metric):
                    OPEN.put((self.G.nodes[child]["level"], child))
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
