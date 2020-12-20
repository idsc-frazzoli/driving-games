from typing import Type, Dict, Mapping, Set
from decimal import Decimal as D

from preferences import Preference, ComparisonOutcome, StrictProductPreferenceDict, SmallerPreferredTol
from .metrics_def import Metric, EvaluatedMetric, PlayerOutcome

__all__ = [
    "EvaluatedMetricPreference",
    "PosetalPreference",
]


class EvaluatedMetricPreference(Preference[EvaluatedMetric]):
    """Compare the total values between evaluated metrics, doesn't check for types"""

    pref: SmallerPreferredTol

    def __init__(self):
        self.pref = SmallerPreferredTol(D("1e-6"))

    def get_type(self) -> Type[EvaluatedMetric]:
        return EvaluatedMetric

    def compare(self, a: EvaluatedMetric, b: EvaluatedMetric) -> ComparisonOutcome:
        return self.pref.compare(a.total, b.total)


# PlayerOutcome = Mapping[Metric, EvaluatedMetric]


class PosetalPreference(Preference[PlayerOutcome]):
    # TODO[SIR]: Using simple product preference for comparison, will change later
    pref: Preference[Mapping[Metric, EvaluatedMetric]]
    keys: Set[Metric]

    def __init__(self, keys: Set[Metric]):
        pref_dict: Dict[Metric, Preference[EvaluatedMetric]] = {p: EvaluatedMetricPreference() for p in keys}
        self.pref = StrictProductPreferenceDict(prefs=pref_dict)
        self.keys = keys

    def get_type(self) -> Type[PlayerOutcome]:
        # fixme az here posetalpreference?
        return PlayerOutcome

    def compare(self, a: PlayerOutcome, b: PlayerOutcome) -> ComparisonOutcome:

        a_keys = set(a.keys())
        b_keys = set(b.keys())
        if not (self.keys == a_keys == b_keys):
            msg = "Mismatch of keys - keys={}, a={}, b={}".format(self.keys, a_keys, b_keys)
            raise ValueError(msg)

        outcome = self.pref.compare(a=a, b=b)
        return outcome
