from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, unsafe_hash=True)
class Equilibrium:
    s1: np.ndarray
    s2: np.ndarray
    p1_payoff: float
    p2_payoff: float

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return (
                self.p1_payoff == other.p1_payoff
                and self.p2_payoff == other.p2_payoff
                and np.array_equal(self.s1, other.s1)
                and np.array_equal(self.s2, other.s2)
        )
