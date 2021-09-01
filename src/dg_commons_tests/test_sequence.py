from decimal import Decimal
from time import process_time

from numpy.testing import assert_raises

from dg_commons import DgSampledSequence, UndefinedAtTime


def test_dg_sampledsequence():
    ts = (1, 2.2, 3, 4, 5)
    tsD = [Decimal(t) for t in ts]
    val = [1, 2, 3, 4, 5]
    t0 = process_time()
    seq = DgSampledSequence[float](ts, values=val)
    t1 = process_time()
    seqD = DgSampledSequence[float](tsD, values=val)

    # at
    with assert_raises(UndefinedAtTime):
        seq.at(4.4)
    atD = Decimal(4)
    at = 8 / 2.0

    assert seq.at(atD) == seq.at(at) == at
    assert seqD.at(atD) == seqD.at(at) == at

    # type, at previous and interp
    for s in [seq, seqD]:
        assert s.XT == float
        assert s.at_or_previous(3.2) == 3
        assert s.at_or_previous(-2) == 1
        assert s.at_or_previous(4.9) == 4
        assert s.at_or_previous(6) == 5
        assert s.at_interp(3.2) == 3.2
        assert s.at_interp(-2) == 1
        assert s.at_interp(3.5) == 3.5
