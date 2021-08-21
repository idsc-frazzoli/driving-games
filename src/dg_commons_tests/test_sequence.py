from decimal import Decimal
from time import process_time

from numpy.testing import assert_raises

from dg_commons import DgSampledSequence, UndefinedAtTime


def test_dg_sampledsequence():
    ts = [1, 2.2, 3, 4, 5]
    tsD = [Decimal(t) for t in ts]
    val = [1, 2, 3, 4, 5]
    seq = DgSampledSequence[float](ts, val)

    with assert_raises(UndefinedAtTime):
        seq.at(4.4)

    t0 = process_time()
    t = DgSampledSequence[float](ts, val)
    t1 = process_time()
    assert t.XT == float
    assert t.at_or_previous(3.2) == 3
    print(t.at_or_previous(-2))
    print(t.at_or_previous(4.9))
    print(t.at_or_previous(6))
    assert t.at_interp(3.2) == 3.2
    print(t.at_interp(-2))
    print(t.at_interp(6))

