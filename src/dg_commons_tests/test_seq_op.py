from decimal import Decimal

from numpy.testing import assert_raises

from dg_commons import DgSampledSequence, seq_accumulate

ts = (1, 2, 3, 4, 5)
tsD = [Decimal(t) for t in ts]
val = [1, 2, 3, 4, 5]
seq = DgSampledSequence[float](ts, values=val)
seqD = DgSampledSequence[float](tsD, values=val)


def test_accumulate():
    expected = (1, 3, 6, 10, 15)
    seq_acc = seq_accumulate(seq)
    seqD_acc = seq_accumulate(seqD)
    assert seq_acc.values == expected
    assert seqD_acc.values == expected


def test_illegal_assign():
    def _try_assign():
        seq.timestamps = [1, -3, 4.5]

    assert_raises(RuntimeError, _try_assign)
