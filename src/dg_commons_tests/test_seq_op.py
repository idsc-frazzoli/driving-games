from numpy.testing import assert_raises

from dg_commons import DgSampledSequence, seq_accumulate


def test_accumulate():
    ts = (1, 2, 3, 4, 5)
    val = [1, 2, 3, 4, 5]
    seq = DgSampledSequence[float](ts, values=val)

    seq2 = seq_accumulate(seq)
    assert seq2.values == [1, 3, 6, 10, 15]

    def _try_assign():
        seq.timestamps = [1, -3, 4.5]
    assert_raises(RuntimeError, _try_assign)
