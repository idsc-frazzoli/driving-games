from operator import add
from typing import List, Tuple

from toolz import accumulate

from dg_commons.sequence import DgSampledSequence, Timestamp, iterate_with_dt

__all__ = ["seq_accumulate"]


# todo check that this operations cannot be done on any values

def seq_integrate(sequence: DgSampledSequence[float]) -> DgSampledSequence[float]:
    """ Integrates with respect to time - multiplies the value with delta T. """
    if not sequence:
        msg = "Cannot integrate empty sequence."
        raise ValueError(msg)
    total = 0.0
    timestamps = []
    values = []
    for _ in iterate_with_dt(sequence):
        v_avg = (_.v0 + _.v1) / 2.0
        total += v_avg * float(_.dt)
        timestamps.append(_.t1)
        values.append(total)

    return DgSampledSequence[float](timestamps, values)


def seq_accumulate(sequence: DgSampledSequence[float]) -> DgSampledSequence[float]:
    cumsum = list(accumulate(add, sequence.values))
    return DgSampledSequence[sequence.XT](sequence.timestamps, cumsum)





