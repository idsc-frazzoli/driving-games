from abc import ABC, abstractmethod
from typing import Callable, Collection, FrozenSet, Generic, Iterable, Tuple, TypeVar

from .poss import Poss, Φ

__all__ = ["PossibilityStructure"]


A = TypeVar("A")
B = TypeVar("B")


class Sampler(Generic[Φ], ABC):
    @abstractmethod
    def sample(self, a: Poss[A, Φ]) -> A:
        """"""


class PossibilityStructure(Generic[Φ], ABC):
    @abstractmethod
    def lift_one(self, a: A) -> Poss[A, Φ]:
        """ From one element to a p.d. on one element """

    @abstractmethod
    def lift_many(self, a: Collection[A]) -> Poss[A, Φ]:
        """ From one element to a p.d. on one element """

    @abstractmethod
    def build(self, a: Poss[A, Φ], f: Callable[[A], B]) -> Poss[B, Φ]:
        raise NotImplementedError()

    @abstractmethod
    def flatten(self, a: Poss[Poss[A, Φ], Φ]) -> Poss[A, Φ]:
        raise NotImplementedError()

    @abstractmethod
    def get_sampler(self, seed: int) -> Sampler[Φ]:
        """"""

    @abstractmethod
    def mix(self, a: Collection[A]) -> FrozenSet[Poss[A, Φ]]:
        """ Return """

    @abstractmethod
    def multiply(self, a: Iterable[Φ]) -> Φ:
        """  """

    @abstractmethod
    def fold(self, a: Iterable[Tuple[A, Φ]]) -> Poss[A, Φ]:
        """  """
