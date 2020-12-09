from dataclasses import dataclass
from decimal import Decimal as D
from typing import Generic, TypeVar, List, Callable, Type, ClassVar, Iterator

__all__ = ["Timestamp", "SampledSequence", "IterateDT", "iterate_with_dt", "UndefinedAtTime"]

X = TypeVar("X")
Y = TypeVar("Y")
Timestamp = D


class UndefinedAtTime(Exception):
    pass


@dataclass
class IterateDT(Generic[Y]):
    t0: Timestamp
    t1: Timestamp
    dt: Timestamp
    v0: Y
    v1: Y


@dataclass
class SampledSequence(Generic[X]):
    """ A sampled time sequence. Only defined at certain points. """

    timestamps: List[Timestamp]
    values: List[X]

    XT: ClassVar[Type[X]] = object

    def __post_init__(self):
        values = list(self.values)
        timestamps = list(self.timestamps)

        if len(timestamps) != len(values):
            msg = "Length mismatch"
            raise ValueError(msg)

        for t in timestamps:
            if not isinstance(t, Timestamp):
                msg = "I expected a real number, got %s" % type(t)
                raise ValueError(msg)
        for i in range(len(timestamps) - 1):
            dt = timestamps[i + 1] - timestamps[i]
            if dt <= 0:
                msg = "Invalid dt = %s at i = %s; ts= %s" % (dt, i, timestamps)
                raise ValueError(msg)
        timestamps = list(map(Timestamp, timestamps))
        self.timestamps = timestamps
        self.values = values

    def at(self, t: Timestamp) -> Generic[X]:
        """Returns value at requested timestamp,
        Raises UndefinedAtTime if not defined at t"""
        try:
            i = self.timestamps.index(t)
        except ValueError:
            msg = "Could not find timestamp %s in %s" % (t, self.timestamps)
            raise UndefinedAtTime(msg)
        else:
            return self.values[i]

    def get_start(self) -> Timestamp:
        """ Returns the timestamp for start """
        if not self.timestamps:
            msg = "Empty sequence"
            raise ValueError(msg)
        return self.timestamps[0]

    def get_end(self) -> Timestamp:
        """ Returns the timestamp for start """
        if not self.timestamps:
            msg = "Empty sequence"
            raise ValueError(msg)
        return self.timestamps[-1]

    def get_sampling_points(self) -> List[Timestamp]:
        """
        Returns the lists of sampled timestamps
        """
        return list(self.timestamps)

    def transform_values(self, f: Callable[[X], Y], YT: Type[Y] = object) -> "SampledSequence[Y]":
        values = []
        timestamps = []
        for t, _ in self:
            res = f(_)
            if res is not None:
                values.append(res)
                timestamps.append(t)

        return SampledSequence[YT](timestamps, values)

    def __iter__(self):
        return zip(self.timestamps, self.values).__iter__()

    def __len__(self) -> int:
        return len(self.timestamps)


def iterate_with_dt(sequence: SampledSequence[X]) -> Iterator[IterateDT[X]]:
    """ yields t0, t1, dt, v0, v1 """
    timestamps = sequence.timestamps
    values = sequence.values
    for i in range(len(timestamps) - 1):
        t0 = timestamps[i]
        assert isinstance(t0, D), type(t0)
        t1 = timestamps[i + 1]
        v0 = values[i]
        v1 = values[i + 1]
        dt = t1 - t0
        X = type(sequence).XT
        yield IterateDT[X](t0, t1, dt, v0, v1)
