from copy import deepcopy
from typing import Set, Dict
from decimal import Decimal as D
from nose.tools import assert_equal

from trajectory_games.preference import PosetalPreference
from driving_games.metrics_structures import Metric, EvaluatedMetric
from dg_commons.seq.sequence import DgSampledSequence

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


def test_poset():
    metrics: Set[Metric] = get_metrics_set()
    pref1 = PosetalPreference(pref_str="test_1", use_cache=False)
    pref2 = PosetalPreference(pref_str="test_2", use_cache=False)
    pref3 = PosetalPreference(pref_str="test_3", use_cache=False)

    default: EvaluatedMetric = EvaluatedMetric(
        value=0.0,
        name="TestMetric",
        pointwise=DgSampledSequence([], []),
    )

    p_def: Dict[Metric, EvaluatedMetric] = {metric: deepcopy(default) for metric in metrics}
    p1 = deepcopy(p_def)
    p2 = deepcopy(p_def)

    # p1==p2
    assert_equal(pref1.compare_old(p1, p2), INDIFFERENT)
    assert_equal(pref2.compare_old(p1, p2), INDIFFERENT)
    assert_equal(pref3.compare_old(p1, p2), INDIFFERENT)

    assert_equal(pref1.compare_old(p1, p2), pref1.compare(p1, p2))
    assert_equal(pref2.compare_old(p1, p2), pref2.compare(p1, p2))
    assert_equal(pref3.compare_old(p1, p2), pref3.compare(p1, p2))

    p2[LongitudinalAcceleration()].value = D("1")
    # LongAcc: p1>p2
    assert_equal(pref1.compare_old(p1, p2), INDIFFERENT)
    assert_equal(pref2.compare_old(p1, p2), FIRST_PREFERRED)
    assert_equal(pref3.compare_old(p1, p2), FIRST_PREFERRED)

    assert_equal(pref1.compare_old(p1, p2), pref1.compare(p1, p2))
    assert_equal(pref2.compare_old(p1, p2), pref2.compare(p1, p2))
    assert_equal(pref3.compare_old(p1, p2), pref3.compare(p1, p2))

    p1[LateralComfort()].value = D("1")
    # LongAcc: p1>p2, LatComf: p1<p2
    assert_equal(pref1.compare_old(p1, p2), SECOND_PREFERRED)
    assert_equal(pref2.compare_old(p1, p2), SECOND_PREFERRED)
    assert_equal(pref3.compare_old(p1, p2), SECOND_PREFERRED)

    assert_equal(pref1.compare_old(p1, p2), pref1.compare(p1, p2))
    assert_equal(pref2.compare_old(p1, p2), pref2.compare(p1, p2))
    assert_equal(pref3.compare_old(p1, p2), pref3.compare(p1, p2))

    p2[MinimumClearance()].value = D("1")
    # LongAcc: p1>p2, LatComf: p1<p2, MinClear: p1>p2
    assert_equal(pref1.compare_old(p1, p2), SECOND_PREFERRED)
    assert_equal(pref2.compare_old(p1, p2), INCOMPARABLE)
    assert_equal(pref3.compare_old(p1, p2), INCOMPARABLE)

    assert_equal(pref1.compare_old(p1, p2), pref1.compare(p1, p2))
    assert_equal(pref2.compare_old(p1, p2), pref2.compare(p1, p2))
    assert_equal(pref3.compare_old(p1, p2), pref3.compare(p1, p2))

    p1[LateralComfort()].value = D("0")
    p2[LongitudinalAcceleration()].value = D("0")
    p1[ProgressAlongReference()].value = D("1")
    # MinClear: p1>p2, Prog: p1<p2
    assert_equal(pref1.compare_old(p1, p2), SECOND_PREFERRED)
    assert_equal(pref2.compare_old(p1, p2), SECOND_PREFERRED)
    assert_equal(pref3.compare_old(p1, p2), SECOND_PREFERRED)

    assert_equal(pref1.compare_old(p1, p2), pref1.compare(p1, p2))
    assert_equal(pref2.compare_old(p1, p2), pref2.compare(p1, p2))
    assert_equal(pref3.compare_old(p1, p2), pref3.compare(p1, p2))

    p1[ProgressAlongReference()].value = D("0")
    # MinClear: p1>p2
    assert_equal(pref1.compare_old(p1, p2), INDIFFERENT)
    assert_equal(pref2.compare_old(p1, p2), FIRST_PREFERRED)
    assert_equal(pref3.compare_old(p1, p2), FIRST_PREFERRED)

    assert_equal(pref1.compare_old(p1, p2), pref1.compare(p1, p2))
    assert_equal(pref2.compare_old(p1, p2), pref2.compare(p1, p2))
    assert_equal(pref3.compare_old(p1, p2), pref3.compare(p1, p2))

    p1[DrivableAreaViolation()].value = D("1")
    # MinClear: p1>p2, Area: p1<p2
    assert_equal(pref1.compare_old(p1, p2), SECOND_PREFERRED)
    assert_equal(pref2.compare_old(p1, p2), SECOND_PREFERRED)
    assert_equal(pref3.compare_old(p1, p2), SECOND_PREFERRED)

    assert_equal(pref1.compare_old(p1, p2), pref1.compare(p1, p2))
    assert_equal(pref2.compare_old(p1, p2), pref2.compare(p1, p2))
    assert_equal(pref3.compare_old(p1, p2), pref3.compare(p1, p2))

    p2[DeviationHeading()].value = D("1")
    # MinClear: p1>p2, Area: p1<p2, DevHead: p1>p2
    assert_equal(pref1.compare_old(p1, p2), SECOND_PREFERRED)
    assert_equal(pref2.compare_old(p1, p2), INCOMPARABLE)
    assert_equal(pref3.compare_old(p1, p2), INCOMPARABLE)

    assert_equal(pref1.compare_old(p1, p2), pref1.compare(p1, p2))
    assert_equal(pref2.compare_old(p1, p2), pref2.compare(p1, p2))
    assert_equal(pref3.compare_old(p1, p2), pref3.compare(p1, p2))

    p1[DeviationLateral()].value = D("1")
    # MinClear: p1>p2, Area: p1<p2, DevHead: p1>p2, DevLat: p1<p2
    assert_equal(pref1.compare_old(p1, p2), SECOND_PREFERRED)
    assert_equal(pref2.compare_old(p1, p2), SECOND_PREFERRED)
    assert_equal(pref3.compare_old(p1, p2), SECOND_PREFERRED)

    assert_equal(pref1.compare_old(p1, p2), pref1.compare(p1, p2))
    assert_equal(pref2.compare_old(p1, p2), pref2.compare(p1, p2))
    assert_equal(pref3.compare_old(p1, p2), pref3.compare(p1, p2))

    p2[CollisionEnergy()].value = D("1")
    # MinClear: p1>p2, Area: p1<p2, DevHead: p1>p2, DevLat: p1<p2, Coll: p1>p2
    assert_equal(pref1.compare_old(p1, p2), FIRST_PREFERRED)
    assert_equal(pref2.compare_old(p1, p2), FIRST_PREFERRED)
    assert_equal(pref3.compare_old(p1, p2), FIRST_PREFERRED)

    assert_equal(pref1.compare_old(p1, p2), pref1.compare(p1, p2))
    assert_equal(pref2.compare_old(p1, p2), pref2.compare(p1, p2))
    assert_equal(pref3.compare_old(p1, p2), pref3.compare(p1, p2))

    p2[MinimumClearance()].value = D("0")
    p1[DrivableAreaViolation()].value = D("0")
    p2[DeviationHeading()].value = D("0")
    p2[CollisionEnergy()].value = D("0")
    # DevLat: p1<p2
    assert_equal(pref1.compare_old(p1, p2), SECOND_PREFERRED)
    assert_equal(pref2.compare_old(p1, p2), SECOND_PREFERRED)
    assert_equal(pref3.compare_old(p1, p2), SECOND_PREFERRED)

    assert_equal(pref1.compare_old(p1, p2), pref1.compare(p1, p2))
    assert_equal(pref2.compare_old(p1, p2), pref2.compare(p1, p2))
    assert_equal(pref3.compare_old(p1, p2), pref3.compare(p1, p2))

    p2[SteeringAngle()].value = D("1")
    # DevLat: p1<p2, StAng: p1>p2
    assert_equal(pref1.compare_old(p1, p2), SECOND_PREFERRED)
    assert_equal(pref2.compare_old(p1, p2), SECOND_PREFERRED)
    assert_equal(pref3.compare_old(p1, p2), INCOMPARABLE)

    assert_equal(pref1.compare_old(p1, p2), pref1.compare(p1, p2))
    assert_equal(pref2.compare_old(p1, p2), pref2.compare(p1, p2))
    assert_equal(pref3.compare_old(p1, p2), pref3.compare(p1, p2))

    p1[EpisodeTime()].value = D("1")
    p2[LongitudinalAcceleration()].value = D("1")
    # DevLat: p1<p2, StAng: p1>p2, Time: p1<p2, LongAcc: p1>p2
    assert_equal(pref1.compare_old(p1, p2), SECOND_PREFERRED)
    assert_equal(pref2.compare_old(p1, p2), SECOND_PREFERRED)
    assert_equal(pref3.compare_old(p1, p2), SECOND_PREFERRED)

    assert_equal(pref1.compare_old(p1, p2), pref1.compare(p1, p2))
    assert_equal(pref2.compare_old(p1, p2), pref2.compare(p1, p2))
    assert_equal(pref3.compare_old(p1, p2), pref3.compare(p1, p2))

    p1[DeviationLateral()].value = D("0")
    p2[SteeringAngle()].value = D("0")
    # Time: p1<p2, LongAcc: p1>p2
    assert_equal(pref1.compare_old(p1, p2), INDIFFERENT)
    assert_equal(pref2.compare_old(p1, p2), FIRST_PREFERRED)
    assert_equal(pref3.compare_old(p1, p2), INCOMPARABLE)

    assert_equal(pref1.compare_old(p1, p2), pref1.compare(p1, p2))
    assert_equal(pref2.compare_old(p1, p2), pref2.compare(p1, p2))
    assert_equal(pref3.compare_old(p1, p2), pref3.compare(p1, p2))

    p1[EpisodeTime()].value = D("0")
    p2[LongitudinalAcceleration()].value = D("0")
    # p1==p2
    assert_equal(pref1.compare_old(p1, p2), INDIFFERENT)
    assert_equal(pref2.compare_old(p1, p2), INDIFFERENT)
    assert_equal(pref3.compare_old(p1, p2), INDIFFERENT)

    assert_equal(pref1.compare_old(p1, p2), pref1.compare(p1, p2))
    assert_equal(pref2.compare_old(p1, p2), pref2.compare(p1, p2))
    assert_equal(pref3.compare_old(p1, p2), pref3.compare(p1, p2))

    return


if __name__ == "__main__":
    test_poset()
