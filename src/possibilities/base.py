from abc import ABC, abstractmethod
from typing import Callable, Collection, FrozenSet, Mapping, TypeVar

from .poss import Poss

__all__ = ["PossibilityMonad"]

A = TypeVar("A")
B = TypeVar("B")
K = TypeVar("K")


class Sampler(ABC):
    """ A Sampler is something that can sample from a distribution. """

    @abstractmethod
    def sample(self, a: Poss[A]) -> A:
        """
        Get one sample from the distribution. Note that a sampler has
        an internal seed, so successive invocations will give different results.
        """


class PossibilityMonad(ABC):
    """
        The interface for a generic uncertainty monad.
    """

    @abstractmethod
    def unit(self, a: A) -> Poss[A]:
        """ Constructs a distribution from one element. The return in Haskell. """

    @abstractmethod
    def lift_many(self, a: Collection[A]) -> Poss[A]:
        """ Constructs a distribution from a set of elements element. """

    @abstractmethod
    def join(self, a: Poss[Poss[A]]) -> Poss[A]:
        """ The flattening operations for a monad. """

    @abstractmethod
    def build(self, a: Poss[A], f: Callable[[A], B]) -> Poss[B]:
        """ Computes the push-forward of a distribution.
        Equivalent to the 'bind' method of monads, with the only difference that the lifting operation
        happens inside build and not f. Indeed 'bind' would require f:Callable[[A], Poss[B]]."""

    @abstractmethod
    def build_multiple(self, a: Mapping[K, Poss[A]], f: Callable[[Mapping[K, A]], B]) -> Poss[B]:
        """ Computes the push-forward from a set of independent distributions. """

    @abstractmethod
    def get_sampler(self, seed: int) -> Sampler:
        """ Creates a sampler object that can be used to simulate realizations from this distribution. """

    @abstractmethod
    def mix(self, support: Collection[A]) -> FrozenSet[Poss[A]]:
        """ Returns a set of "distributions" with the given support. """


# V = TypeVar('V')
# Z = TypeVar('Z')
#
# def compute_marginals(ps: PossibilityMonad, a: Poss[Mapping[K, V]],
#               f: Callable[[K, V], Z]) -> Mapping[K, Poss[Z]]:
#     pass
