import random
from dataclasses import dataclass
from typing import Union, Callable
import math
import scipy
import numpy as np


class Empty:
    pass


@dataclass
class LGBParam:
    failure_p: float = 0
    """ Failure Probability """

    def __post_init__(self):
        assert 1 >= self.failure_p >= 0


class LGB:
    def __init__(self, params=LGBParam()):
        self.params = params
        self.steps = 0
        self.counter = 0

    def drop(self) -> bool:
        val = random.uniform(0, 1)

        return_val = False
        if val <= self.params.failure_p:
            return_val = True

        current_state = 0 if return_val else 1
        self.counter += current_state
        self.steps += 1

        return return_val

    def mean(self):
        return self.counter/self.steps


@dataclass
class LGMParam(LGBParam):
    recovery_p: float = 0
    """ Recovery Probability """
    def __post_init__(self):
        super().__post_init__()
        assert 1 >= self.recovery_p >= 0

        mat = np.array([[1-self.failure_p, self.failure_p], [self.recovery_p, 1-self.recovery_p]])
        eigval, eigvecl, eigvecr = scipy.linalg.eig(mat, left=True)

        steady_mat = np.matmul(np.matmul(eigvecl, np.diag([0 if eig < 1 else 1 for eig in eigval])),
                               scipy.linalg.inv(eigvecl))
        steady_values = np.matmul(steady_mat, np.array([[1], [0]]))
        self.expected_value = steady_values[0]*1 + 0*steady_values[1]


class LGM:
    def __init__(self, params=LGMParam()):
        self.params = params
        self.current_state = 1
        self.counter = 0
        self.steps = 0

    def drop(self) -> bool:
        val = random.uniform(0, 1)

        return_val = False
        if self.current_state == 1:
            if val <= self.params.failure_p:
                self.current_state = 0
                return_val = True
        else:
            if val <= self.params.recovery_p:
                self.current_state = 1
            else:
                return_val = True

        self.counter += self.current_state
        self.steps += 1

        return return_val

    def mean(self):
        return self.counter/self.steps


@dataclass
class ExponentialParams:
    lamb: float = 1

    def __post_init__(self):
        assert self.lamb > 0


class Exponential:
    def __init__(self, params: ExponentialParams):
        self.params = params

    def cdf(self, t):
        assert t >= 0

        return 1-math.exp(-self.params.lamb*t)

    def pdf(self, t):
        assert t >= 0

        return self.params.lamb*math.exp(-self.params.lamb*t)


PDistribution = Union[Empty, Exponential]
PDistributionParams = Union[Empty, ExponentialParams]


@dataclass
class LGSMParam:
    failure_distribution: type(PDistribution) = Exponential
    failure_params: PDistributionParams = ExponentialParams()
    recovery_distribution: type(PDistribution) = Exponential
    recovery_params: PDistributionParams = ExponentialParams()
    dt: float = 0.1


class LGSM:
    def __init__(self, params=LGSMParam()):
        self.dt = params.dt
        self.failure_distribution = params.failure_distribution(params.failure_params)
        self.recovery_distribution = params.recovery_distribution(params.recovery_params)

        self.failure_deltas = []
        self.recovery_deltas = []
        self.current_state = 1
        self.delta_t = 0
        self.val = random.uniform(0, 1)
        self.counter = 0
        self.steps = 0

    def drop(self) -> bool:

        return_val = False
        if self.current_state == 1:
            if self.change_from_1():
                self.failure_deltas.append(self.delta_t-self.dt/2)
                self.delta_t = 0
                self.current_state = 0
                self.val = random.uniform(0, 1)
                return_val = True
        elif self.current_state == 0:
            if self.change_from_0():
                self.recovery_deltas.append(self.delta_t-self.dt/2)
                self.delta_t = 0
                self.current_state = 1
                self.val = random.uniform(0, 1)
            else:
                return_val = True
        self.delta_t += self.dt

        self.counter += self.current_state
        self.steps += 1

        return return_val

    def mean(self):
        return self.counter/self.steps, \
               sum(self.failure_deltas) / len(self.failure_deltas) if len(self.failure_deltas) != 0 else None, \
               sum(self.recovery_deltas) / len(self.recovery_deltas) if len(self.recovery_deltas) != 0 else None

    def change_from_1(self) -> bool:
        val = self.failure_distribution.cdf(self.delta_t)
        return self.val <= val

    def change_from_0(self) -> bool:
        val = self.recovery_distribution.cdf(self.delta_t)
        return self.val <= val


DroppingTechniques = Union[Empty, LGB, LGM, LGSM]
DroppingTechniquesParams = Union[Empty, LGBParam, LGMParam, LGSMParam]
