
from dg_commons import DgSampledSequence, seq_accumulate


def test_accumulate():
    ts = (1, 2, 3, 4, 5)
    val = [1, 2, 3, 4, 5]
    seq = DgSampledSequence[float](ts, values=val)

    seq2 = seq_accumulate(seq)
    print(seq2)
    seq.timestamps = [1, -3, 4.5]
    print(seq)
    print(seq2)
