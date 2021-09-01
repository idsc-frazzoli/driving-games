from decimal import Decimal as D
from time import process_time

from numpy.testing import assert_raises
from zuper_commons.types import ZValueError

from dg_commons import DgSampledSequence, UndefinedAtTime


def test_dg_sampledsequence():
    ts = (1, 2.2, 3, 4, 5)
    tsD = [D(t) for t in ts]
    val = [1, 2, 3, 4, 5]
    t0 = process_time()
    seq = DgSampledSequence[float](ts, values=val)
    t1 = process_time()
    seqD = DgSampledSequence[float](tsD, values=val)

    # at
    with assert_raises(UndefinedAtTime):
        seq.at(4.4)
    atD = D(4)
    at = 8 / 2.0

    assert seq.at(atD) == seq.at(at) == at
    assert seqD.at(atD) == seqD.at(at) == at

    # type, at previous and interp
    for s in [seq, seqD]:
        assert s.XT == float
        assert s.at_or_previous(3.2) == 3
        assert s.at_or_previous(D(3.2)) == 3
        assert s.at_or_previous(-2) == 1
        assert s.at_or_previous(D(-2)) == 1
        assert s.at_or_previous(4.9) == 4
        assert s.at_or_previous(D(4.9)) == 4
        assert s.at_or_previous(6) == 5
        assert s.at_or_previous(D(6)) == 5
        assert s.at_interp(3.2) == 3.2
        assert s.at_interp(D(3.2)) == 3.2
        assert s.at_interp(-2) == 1
        assert s.at_interp(D(-2)) == 1
        assert s.at_interp(3.5) == 3.5
        assert s.at_interp(D(3.5)) == 3.5

    def illegal_timestamps():
        ts = [D(0), 1]
        v = [1, 2]
        _ = DgSampledSequence[float](timestamps=ts, values=v)

    assert_raises(ZValueError, illegal_timestamps)
