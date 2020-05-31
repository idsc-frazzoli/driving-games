from abc import ABC, abstractmethod
from typing import FrozenSet, Generic, Type, TypeVar

from zuper_commons.types import check_isinstance, ZValueError

__all__ = ["Poss", "check_poss"]

A = TypeVar("A")


class Poss(Generic[A], ABC):
    @abstractmethod
    def check_contains(self, T: type, **kwargs):
        pass

    @abstractmethod
    def support(self) -> FrozenSet[A]:
        pass


CHECK = True


def check_poss(a: Poss[A], T: Type[A] = object, **kwargs):
    if not CHECK:
        return
    check_isinstance(a, Poss, **kwargs)
    if T is not object:
        for _ in a.support():
            if not isinstance(_, T):
                raise ZValueError(_=_, T=T, **kwargs)


#
# def check_poss_single(a: Poss[A, Φ], T: Type[A] = object, **kwargs):
#     if not CHECK:
#         return
#     check_isinstance(a, Poss, **kwargs)
#     for _ in a.support():
#         if isinstance(_, Poss):
#             raise ZValueError(_=_, **kwargs)
#     if T is not object:
#         for _ in a.support():
#             if not isinstance(_, T):
#                 raise ZValueError(_=_, T=T, **kwargs)
