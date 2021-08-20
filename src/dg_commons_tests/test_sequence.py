from dg_commons import DgSampledSequence


def test_accumulate():
    ts = [1, 2, 3, 4, 5]
    val = [1, 2, 3, 4, 5]
    seq = DgSampledSequence[float](ts,val)
