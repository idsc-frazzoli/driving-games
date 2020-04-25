from typing import FrozenSet as ASet, Mapping, TypeVar

from frozendict import frozendict

from zuper_commons.types import check_isinstance, ZNotImplementedError
from zuper_typing import debug_print
from .poset import COMP_OUTCOMES, ComparisonOutcome, FIRST_PREFERRED, P, Preference

__all__ = ["SetPreference1"]


class SetPreference1(Preference[ASet[P]]):
    p0: Preference[P]

    def __init__(self, p0: Preference[P]):
        self.p0 = p0

    def __repr__(self):
        d = {'p0': self.p0}
        return 'SetPreference1: ' + debug_print(d)

    def compare(self, a: ASet[P], b: ASet[P]) -> ComparisonOutcome:
        check_isinstance(a, set)
        check_isinstance(b, set)
        if len(a) == 1 and len(b) == 1:
            a1 = list(a)[0]
            b1 = list(b)[0]
            res = self.p0.compare(a1, b1)
            assert res in COMP_OUTCOMES, (res, self.p0)
            return res
        raise ZNotImplementedError(a=a, b=b)


A = TypeVar('A')
B = TypeVar('B')


def remove_dominated(orig: Mapping[A, B], pref: Preference[B]) -> Mapping[A, B]:
    def is_strictly_dominated(x: B) -> bool:
        for k1, v1 in orig.items():
            res = pref.compare(v1, x)
            if res == FIRST_PREFERRED:
                return True
        return False

    res = {}
    for k, v in orig.items():
        if not is_strictly_dominated(v):
            res[k] = v
    return frozendict(res)
