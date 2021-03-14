from copy import deepcopy
from typing import Set, Dict
from decimal import Decimal as D
from nose.tools import assert_equal

from trajectory_games import PosetalPreference, Metric, EvaluatedMetric, SampledSequence

from trajectory_games.metrics import (
    get_metrics_set,
    EpisodeTime,
    DeviationLateral,
    DeviationHeading,
    DrivableAreaViolation,
    ProgressAlongReference,
    LongitudinalAcceleration,
    LongitudinalJerk,
    LateralComfort,
    SteeringAngle,
    SteeringRate,
    CollisionEnergy,
)

from preferences import INDIFFERENT, INCOMPARABLE, FIRST_PREFERRED, SECOND_PREFERRED


def test_poset():
    metrics: Set[Metric] = get_metrics_set()
    pref1 = PosetalPreference(pref_str="test_1")
    pref2 = PosetalPreference(pref_str="test_2")
    pref3 = PosetalPreference(pref_str="test_3")

    default: EvaluatedMetric = EvaluatedMetric(
        total=D("0"),
        description="",
        title="",
        incremental=SampledSequence([], []),
        cumulative=SampledSequence([], []),
    )

    p_def: Dict[Metric, EvaluatedMetric] = {metric: deepcopy(default) for metric in metrics}
    p1 = deepcopy(p_def)
    p2 = deepcopy(p_def)

    # p1==p2
    assert_equal(pref1.compare(p1, p2), INDIFFERENT)
    assert_equal(pref2.compare(p1, p2), INDIFFERENT)
    assert_equal(pref3.compare(p1, p2), INDIFFERENT)

    p2[LongitudinalAcceleration()].total = D("1")
    # LongAcc: p1>p2
    assert_equal(pref1.compare(p1, p2), INDIFFERENT)
    assert_equal(pref2.compare(p1, p2), FIRST_PREFERRED)
    assert_equal(pref3.compare(p1, p2), FIRST_PREFERRED)

    p1[LateralComfort()].total = D("1")
    # LongAcc: p1>p2, LatComf: p1<p2
    assert_equal(pref1.compare(p1, p2), SECOND_PREFERRED)
    assert_equal(pref2.compare(p1, p2), SECOND_PREFERRED)
    assert_equal(pref3.compare(p1, p2), SECOND_PREFERRED)

    p2[LongitudinalJerk()].total = D("1")
    # LongAcc: p1>p2, LatComf: p1<p2, LongJerk: p1>p2
    assert_equal(pref1.compare(p1, p2), SECOND_PREFERRED)
    assert_equal(pref2.compare(p1, p2), INCOMPARABLE)
    assert_equal(pref3.compare(p1, p2), INCOMPARABLE)

    p1[LateralComfort()].total = D("0")
    p2[LongitudinalAcceleration()].total = D("0")
    p1[ProgressAlongReference()].total = D("1")
    # LongJerk: p1>p2, Prog: p1<p2
    assert_equal(pref1.compare(p1, p2), SECOND_PREFERRED)
    assert_equal(pref2.compare(p1, p2), SECOND_PREFERRED)
    assert_equal(pref3.compare(p1, p2), SECOND_PREFERRED)

    p1[ProgressAlongReference()].total = D("0")
    # LongJerk: p1>p2
    assert_equal(pref1.compare(p1, p2), INDIFFERENT)
    assert_equal(pref2.compare(p1, p2), FIRST_PREFERRED)
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

    p2[LongitudinalJerk()].total = D("0")
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
