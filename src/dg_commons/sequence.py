from bisect import bisect_right
from dataclasses import dataclass, InitVar, field
from decimal import Decimal as D
from typing import Generic, TypeVar, List, Callable, Type, ClassVar, Iterator, Union, get_args, Any, Sequence

from zuper_commons.types import ZException, ZValueError

__all__ = ["Timestamp", "DgSampledSequence", "IterateDT", "iterate_with_dt", "UndefinedAtTime"]

X = TypeVar("X")
Y = TypeVar("Y")
Timestamp = Union[D, float, int]


class UndefinedAtTime(ZException):
    pass


@dataclass
class IterateDT(Generic[X]):
    t0: Timestamp
    t1: Timestamp
    dt: Timestamp
    v0: X
    v1: X


@dataclass
class DgSampledSequence(Generic[X]):
    """ A sampled time sequence. Only defined at certain points.
    Modernized version of the original SampledSequence from Duckietown:
    https://github.com/duckietown/duckietown-world/blob/daffy/src/duckietown_world/seqs/tsequence.py
    Modification:
        - Adds the possibility of interpolating
        - removing possibility of assigning post-init timestamps and values fields
    """
    timestamps: InitVar[Sequence[Timestamp]]
    values: InitVar[Sequence[X]]

    _timestamps: List[Timestamp] = field(default_factory=list)
    _values: List[X] = field(default_factory=list)

    XT: ClassVar[Type[X]] = field(init=False)

    def __post_init__(self, timestamps, values):
        timestamps = list(timestamps)
        values = list(values)

        if len(timestamps) != len(values):
            raise ZValueError("Length mismatch of Timestamps and values")

        for t in timestamps:
            if not isinstance(t, get_args(Timestamp)):
                raise ZValueError(f"I expected a real number as \"Timestamp\", got {type(t)}")
        for i in range(len(timestamps) - 1):
            dt = timestamps[i + 1] - timestamps[i]
            if dt <= 0:
                raise ZValueError(f"Invalid dt = {dt} at i = {i}; ts= {timestamps}")
        ts_types = set([type(ts) for ts in self._timestamps])
        if not len(ts_types) == 1:
            # fixme can be controversial
            self._timestamps = list(map(float, timestamps))
        else:
            self._timestamps = timestamps
        self._values = values

    @property
    def timestamps(self) -> List[Timestamp]:
        return self._timestamps

    @timestamps.setter
    def timestamps(self, v: Any) -> None:
        raise RuntimeError("Cannot set timestamps of SampledSequence directly")

    @property
    def values(self) -> List[X]:
        return self._values

    @values.setter
    def values(self, v: Any) -> None:
        raise RuntimeError("Cannot set timestamps of SampledSequence directly")

    def at(self, t: Timestamp) -> X:
        """ Returns value at requested timestamp, raises UndefinedAtTime if not defined at t"""
        try:
            i = self._timestamps.index(t)
            return self._values[i]
        except ValueError:
            msg = f"Could not find Timestamp: {t} in: {self._timestamps}"
            raise UndefinedAtTime(msg)

    def at_or_previous(self, t: Timestamp) -> X:
        try:
            return self.at(t)
        except UndefinedAtTime:
            pass

        last_i = 0
        for i in range(len(self._timestamps)):
            if self._timestamps[i] < t:
                last_i = i
            else:
                break
        return self._values[last_i]

    def get_start(self) -> Timestamp:
        """ Returns the timestamp for start """
        if not self._timestamps:
            raise ZValueError("Empty sequence")
        return self._timestamps[0]

    def get_end(self) -> Timestamp:
        """ Returns the timestamp for start """
        if not self._timestamps:
            raise ZValueError("Empty sequence")
        return self._timestamps[-1]

    def get_sampling_points(self) -> List[Timestamp]:
        """
        Returns the lists of sampled timestamps
        """
        return self._timestamps

    def transform_values(self, f: Callable[[X], Y], YT: Type[Y] = object) -> "DgSampledSequence[Y]":
        values = []
        timestamps = []
        for t, _ in self:
            res = f(_)
            if res is not None:
                values.append(res)
                timestamps.append(t)
        return DgSampledSequence[YT](timestamps, values)

    def get_interp(self, t: Timestamp) -> Generic[X]:
        """ Returns value at requested timestamp,
        Interpolates between timestamps, holds at the extremes"""
        if t <= self.get_start():
            return self.at(self.get_start())
        elif t >= self.get_end():
            return self.at(self.get_end())
        else:
            i = bisect_right(self._timestamps, t)
            scale = float((t - self._timestamps[i - 1]) /
                          (self._timestamps[i] - self._timestamps[i - 1]))
            return self._values[i - 1] * (1 - scale) + self._values[i] * scale

    def __iter__(self):
        return zip(self._timestamps, self._values).__iter__()

    def __len__(self) -> int:
        return len(self._timestamps)


def iterate_with_dt(sequence: DgSampledSequence[X]) -> Iterator[IterateDT[X]]:
    """ yields t0, t1, dt, v0, v1 """
    timestamps = sequence.timestamps
    values = sequence.values
    for i in range(len(timestamps) - 1):
        t0 = timestamps[i]
        assert isinstance(t0, get_args(Timestamp)), type(t0)
        t1 = timestamps[i + 1]
        v0 = values[i]
        v1 = values[i + 1]
        dt = t1 - t0
        X = type(sequence).XT
        yield IterateDT[X](t0, t1, dt, v0, v1)


@dataclass
class DgSampledSequenceBuilder(Generic[X]):
    timestamps: List[Timestamp] = field(default_factory=list)
    values: List[X] = field(default_factory=list)

    XT: ClassVar[Type[X]] = Any

    def add(self, t: Timestamp, v: X):
        if self.timestamps:
            if t == self.timestamps[-1]:
                msg = "Repeated time stamp"
                raise ZValueError(msg, t=t, timestamps=self.timestamps)
        self.timestamps.append(t)
        self.values.append(v)

    def __len__(self) -> int:
        return len(self.timestamps)

    def as_sequence(self) -> DgSampledSequence:
        return DgSampledSequence[self.XT](timestamps=self.timestamps, values=self.values)
