from abc import ABC, abstractmethod
from typing import Callable, Collection, FrozenSet, Generic, Mapping, TypeVar

from .poss import Poss, Φ

__all__ = ["PossibilityStructure"]

A = TypeVar("A")
B = TypeVar("B")
K = TypeVar("K")


class Sampler(Generic[Φ], ABC):
    """ A Sampler is something that can sample from a distribution. """

    @abstractmethod
    def sample(self, a: Poss[A, Φ]) -> A:
        """
        Get one sample from the distribution. Note that a sampler has
        an internal seed, so successive invocations will give different results.
        """


class PossibilityStructure(Generic[Φ], ABC):
    """
        The interface for a generic uncertainty monad.

    """

    @abstractmethod
    def lift_one(self, a: A) -> Poss[A, Φ]:
        """ Constructs a distribution from one element. """

    @abstractmethod
    def lift_many(self, a: Collection[A]) -> Poss[A, Φ]:
        """ Constructs a distribution from a set of elements element. """

    @abstractmethod
    def flatten(self, a: Poss[Poss[A, Φ], Φ]) -> Poss[A, Φ]:
        """ The flattening operations for a monad. """

    @abstractmethod
    def build(self, a: Poss[A, Φ], f: Callable[[A], B]) -> Poss[B, Φ]:
        """ Computes the push-forward of a distribution. """

    @abstractmethod
    def build_multiple(self, a: Mapping[K, Poss[A, Φ]], f: Callable[[Mapping[K, A]], B]) -> Poss[B, Φ]:
        """ Computes the push-forward from a set of independent distributions. """

    @abstractmethod
    def get_sampler(self, seed: int) -> Sampler[Φ]:
        """ Creates a sampler object that can be used to simulate realizations from this distribution. """

    @abstractmethod
    def mix(self, support: Collection[A]) -> FrozenSet[Poss[A, Φ]]:
        """ Returns a set of "distributions" with the given support. """
