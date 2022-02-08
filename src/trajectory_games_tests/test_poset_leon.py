import itertools
from copy import deepcopy
from typing import Set, Dict, Tuple
from decimal import Decimal as D

from networkx import DiGraph, topological_sort, has_path
from nose.tools import assert_equal

from trajectory_games import PosetalPreference, Metric, EvaluatedMetric, DgSampledSequence, WeightedPreference

from trajectory_games.metrics import (
    get_metrics_set,
    EpisodeTime,
    DeviationLateral,
    DeviationHeading,
    DrivableAreaViolation,
    ProgressAlongReference,
    LongitudinalAcceleration,
    LateralComfort,
    SteeringAngle,
    SteeringRate,
    CollisionEnergy,
    MinimumClearance,
)

from preferences import INDIFFERENT, INCOMPARABLE, FIRST_PREFERRED, SECOND_PREFERRED, ComparisonOutcome


def test_poset_leon():
    metrics: Set[Metric] = get_metrics_set()
    pref1 = PosetalPreference(pref_str="pref_granny_level_1", use_cache=False)
    #pref2 = PosetalPreference(pref_str="common_person_2", use_cache=False)
    pref3 = PosetalPreference(pref_str="pref_ambulance", use_cache=False)

    default: EvaluatedMetric = EvaluatedMetric(
        total=0.0,
        description="",
        title="",
        incremental=DgSampledSequence([], []),
        cumulative=DgSampledSequence([], []),
    )

    p_def: Dict[Metric, EvaluatedMetric] = {metric: deepcopy(default) for metric in metrics}
    p1 = deepcopy(p_def)
    p2 = deepcopy(p_def)

    # p1==p2
    assert_equal(pref1.compare(p1, p2), INDIFFERENT)
    #assert_equal(pref2.compare(p1, p2), INDIFFERENT)
    assert_equal(pref3.compare(p1, p2), INDIFFERENT)

    p2[CollisionEnergy()].total = D("1")
    # CollEn: p1>p2
    assert_equal(pref1.compare(p1, p2), FIRST_PREFERRED)
    #assert_equal(pref2.compare(p1, p2), FIRST_PREFERRED)
    assert_equal(pref3.compare(p1, p2), FIRST_PREFERRED)

    p2[CollisionEnergy()].total = D("0")
    p1[LateralComfort()].total = D("1")
    p2[MinimumClearance()].total = D("1")
    # MinClear: p1>p2, LatComf: p1<p2
    assert_equal(pref1.compare(p1, p2), INCOMPARABLE)
    #assert_equal(pref2.compare(p1, p2), INDIFFERENT)
    assert_equal(pref3.compare(p1, p2), FIRST_PREFERRED)

    p2[DeviationLateral()].total = D("1")
    # MinClear: p1>p2, LatComf: p1<p2, DevLat: p1>p2
    assert_equal(pref1.compare(p1, p2), INCOMPARABLE) # WHY IS THIS INCOMPARABLE?
    #assert_equal(pref2.compare(p1, p2), FIRST_PREFERRED)
    assert_equal(pref3.compare(p1, p2), FIRST_PREFERRED)

    p2[DeviationLateral()].total = D("0")
    p2[DeviationHeading()].total = D("1")
    # MinClear: p1>p2, LatComf: p1<p2, DevHead: p1>p2
    assert_equal(pref1.compare(p1, p2), INCOMPARABLE) # WHY IS THIS INCOMPARABLE?
    #assert_equal(pref2.compare(p1, p2), FIRST_PREFERRED)
    assert_equal(pref3.compare(p1, p2), FIRST_PREFERRED)





    p1[LateralComfort()].total = D("0")
    p2[LongitudinalAcceleration()].total = D("0")
    p1[ProgressAlongReference()].total = D("1")
    # MinClear: p1>p2, Prog: p1<p2
    assert_equal(pref1.compare(p1, p2), SECOND_PREFERRED)
    #assert_equal(pref2.compare(p1, p2), SECOND_PREFERRED)
    assert_equal(pref3.compare(p1, p2), SECOND_PREFERRED)

    p1[ProgressAlongReference()].total = D("0")
    # LongJerk: p1>p2
    assert_equal(pref1.compare(p1, p2), INDIFFERENT)
    #assert_equal(pref2.compare(p1, p2), FIRST_PREFERRED)
    assert_equal(pref3.compare(p1, p2), FIRST_PREFERRED)

    p1[DrivableAreaViolation()].total = D("1")
    # LongJerk: p1>p2, Area: p1<p2
    assert_equal(pref1.compare(p1, p2), SECOND_PREFERRED)
    assert_equal(pref2.compare(p1, p2), SECOND_PREFERRED)
    assert_equal(pref3.compare(p1, p2), SECOND_PREFERRED)

    p2[DeviationHeading()].total = D("1")
    # LongJerk: p1>p2, Area: p1<p2, DevHead: p1>p2
    assert_equal(pref1.compare(p1, p2), SECOND_PREFERRED)
    assert_equal(pref2.compare(p1, p2), INCOMPARABLE)
    assert_equal(pref3.compare(p1, p2), INCOMPARABLE)

    p1[DeviationLateral()].total = D("1")
    # LongJerk: p1>p2, Area: p1<p2, DevHead: p1>p2, DevLat: p1<p2
    assert_equal(pref1.compare(p1, p2), SECOND_PREFERRED)
    assert_equal(pref2.compare(p1, p2), SECOND_PREFERRED)
    assert_equal(pref3.compare(p1, p2), SECOND_PREFERRED)

    p2[CollisionEnergy()].total = D("1")
    # LongJerk: p1>p2, Area: p1<p2, DevHead: p1>p2, DevLat: p1<p2, Coll: p1>p2
    assert_equal(pref1.compare(p1, p2), FIRST_PREFERRED)
    assert_equal(pref2.compare(p1, p2), FIRST_PREFERRED)
    assert_equal(pref3.compare(p1, p2), FIRST_PREFERRED)

    p2[MinimumClearance()].total = D("0")
    p1[DrivableAreaViolation()].total = D("0")
    p2[DeviationHeading()].total = D("0")
    p2[CollisionEnergy()].total = D("0")
    # DevLat: p1<p2
    assert_equal(pref1.compare(p1, p2), SECOND_PREFERRED)
    assert_equal(pref2.compare(p1, p2), SECOND_PREFERRED)
    assert_equal(pref3.compare(p1, p2), SECOND_PREFERRED)

    p2[SteeringAngle()].total = D("1")
    # DevLat: p1<p2, StAng: p1>p2
    assert_equal(pref1.compare(p1, p2), SECOND_PREFERRED)
    assert_equal(pref2.compare(p1, p2), SECOND_PREFERRED)
    assert_equal(pref3.compare(p1, p2), INCOMPARABLE)

    p1[EpisodeTime()].total = D("1")
    p2[LongitudinalAcceleration()].total = D("1")
    # DevLat: p1<p2, StAng: p1>p2, Surv: p1<p2, LongAcc: p1>p2
    assert_equal(pref1.compare(p1, p2), SECOND_PREFERRED)
    assert_equal(pref2.compare(p1, p2), SECOND_PREFERRED)
    assert_equal(pref3.compare(p1, p2), SECOND_PREFERRED)

    p1[DeviationLateral()].total = D("0")
    p2[SteeringAngle()].total = D("0")
    # Surv: p1<p2, LongAcc: p1>p2
    assert_equal(pref1.compare(p1, p2), INDIFFERENT)
    assert_equal(pref2.compare(p1, p2), FIRST_PREFERRED)
    assert_equal(pref3.compare(p1, p2), INCOMPARABLE)

    p1[EpisodeTime()].total = D("0")
    p2[LongitudinalAcceleration()].total = D("0")
    # p1==p2
    assert_equal(pref1.compare(p1, p2), INDIFFERENT)
    assert_equal(pref2.compare(p1, p2), INDIFFERENT)
    assert_equal(pref3.compare(p1, p2), INDIFFERENT)


CompareDict: Dict[Tuple[bool, bool], ComparisonOutcome] \
    = {(False, False): INCOMPARABLE, (True, False): FIRST_PREFERRED,
       (False, True): SECOND_PREFERRED, (True, True): INDIFFERENT}


def compare_posets(A: PosetalPreference, B: PosetalPreference) -> ComparisonOutcome:
    first = check_subset(A, B)
    second = check_subset(B, A)
    return CompareDict[(first, second)]


def check_subset(A: PosetalPreference, B: PosetalPreference) -> bool:

    # Check if node is a weighted node or not
    def check_weighted(wnode: WeightedPreference) -> bool:
        total: int = sum([w > D("0") for _, w in wnode.weights.items()])
        return total != 1

    # Save nodes from preference to local graph
    def add_nodes(pref: PosetalPreference) -> Tuple[DiGraph, Set[str]]:
        nodes = DiGraph()
        wnodes: Set[str] = set()
        for wnode in topological_sort(G=pref.graph):
            nodes.add_node(wnode.name)
            for pred in pref.graph.predecessors(wnode):
                nodes.add_edge(u_of_edge=pred.name, v_of_edge=wnode.name)
            if check_weighted(wnode=wnode): wnodes.add(wnode.name)
        return nodes, wnodes

    graph_a, wnodes_a = add_nodes(pref=A)
    graph_b, wnodes_b = add_nodes(pref=B)

    if not wnodes_b.issubset(wnodes_a):
        # Find predecessors and successors for each of the nodes in b, but not in a
        for node in wnodes_b.difference(wnodes_a):
            # w_pref = PosetalPreference._node_dict[node]
            pred = list(graph_b.predecessors(node))
            succ = list(graph_b.successors(node))
            graph_b.remove_node(node)

            # Replace the weighted node by each of it's constituents (incomp)
            for metric, weight in PosetalPreference._node_dict[node].weights.items():
                if weight <= D("0"): continue
                mname = type(metric).__name__
                if graph_b.has_node(mname):
                    raise AssertionError(f"Graph already has node {mname}.\n"
                                         f"All nodes = {graph_b.nodes}")
                graph_b.add_node(mname)
                edges = [(p, mname) for p in pred] + [(mname, s) for s in succ]
                graph_b.add_edges_from(edges)

    # If nodes in A is not a subset of nodes in B, A can't be subset of B
    nodes_a, nodes_b = set(graph_a.nodes), set(graph_b.nodes)
    if not nodes_a.issubset(nodes_b):
        return False

    # All nodes in B, but not A should be lexicographic to all nodes in A
    if not all([has_path(G=graph_b, source=a_node, target=b_node)
                for a_node, b_node in
                itertools.product(nodes_a, nodes_b.difference(nodes_a))]):
        return False

    # Ensure all lexi in A are still lexi in B
    sorted_a = list(topological_sort(G=graph_a))
    for i in range(0, len(sorted_a)-1):
        i_node = sorted_a[i]
        for j in range(i, len(sorted_a)):
            j_node = sorted_a[j]
            if has_path(G=graph_a, source=i_node, target=j_node) and \
                    not has_path(G=graph_b, source=i_node, target=j_node):
                return False

    return True


def test_compare_posets():

    # Initialise prefs (pref[0] is empty)
    prefs = [PosetalPreference(pref_str="NoPreference", use_cache=False)] +\
            [PosetalPreference(pref_str=f"comp_{i}", use_cache=False) for i in range(1, 15)]

    results: Dict[Tuple[int, int], ComparisonOutcome] = {
        (1, 2): INCOMPARABLE,
        (1, 3): FIRST_PREFERRED,
        (1, 4): INCOMPARABLE,
        (1, 5): INCOMPARABLE,
        (1, 6): INCOMPARABLE,
        (1, 7): FIRST_PREFERRED,
        (2, 3): FIRST_PREFERRED,
        (2, 4): INCOMPARABLE,
        (2, 5): INCOMPARABLE,
        (2, 6): FIRST_PREFERRED,
        (2, 7): INCOMPARABLE,
        (3, 4): INCOMPARABLE,
        (3, 5): INCOMPARABLE,
        (3, 6): INCOMPARABLE,
        (3, 7): INCOMPARABLE,
        (4, 5): FIRST_PREFERRED,
        (4, 6): FIRST_PREFERRED,
        (4, 7): FIRST_PREFERRED,
        (5, 6): FIRST_PREFERRED,
        (5, 7): FIRST_PREFERRED,
        (6, 7): INCOMPARABLE,

        (1, 8): FIRST_PREFERRED,
        (4, 8): FIRST_PREFERRED,
        (7, 8): FIRST_PREFERRED,
        (3, 9): FIRST_PREFERRED,
        (7, 9): FIRST_PREFERRED,
        (3, 10): FIRST_PREFERRED,

        (9, 11): SECOND_PREFERRED,
        (10, 11): SECOND_PREFERRED,
        (7, 12): FIRST_PREFERRED,
        (9, 12): SECOND_PREFERRED,
        (10, 12): INCOMPARABLE,
        (12, 13): FIRST_PREFERRED,
        (8, 13): FIRST_PREFERRED,

    }

    for (A, B), result in results.items():
        ab = compare_posets(prefs[A], prefs[B])
        assert_equal(ab, result, f"A,B = {A, B}\t res,actual = {ab, result}")


if __name__ == '__main__':
    test_poset_leon()
