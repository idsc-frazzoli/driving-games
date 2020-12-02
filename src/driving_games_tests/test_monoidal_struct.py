from decimal import Decimal as D, Decimal
from fractions import Fraction
from functools import reduce
from operator import add
from time import perf_counter

from driving_games import VehicleCosts, Dict


def test_decimal_fraction():
    f1 = Fraction(1, 3)
    print(f1)
    f2 = D(1 / 3)
    print(f2)
    d1 = D(1)
    print(d1 * D(float(f1)))
    print(d1 * D(float(f2)))

    tic = perf_counter()
    cost_vector = [VehicleCosts(D(i)) for i in range(10000)]
    cost_vector_sum = reduce(add, cost_vector)
    elapsed = perf_counter() - tic
    print(f"result is {cost_vector_sum}, elapsed in {elapsed}")

    class DurationCosts(Decimal):
        # support weight multiplication for expected value

        def __mul__(self, weight: Fraction) -> "DurationCosts":
            # weighting costs, e.g. according to a probability
            return DurationCosts(self * D(float(weight)))

        __rmul__ = __mul__

        def __add__(self, other):
            return DurationCosts(super(DurationCosts, self).__add__(other))

        __radd__ = __add__

        def __repr__(self):
            return f"DurationCost:('{self.__str__()}')"

    tic = perf_counter()
    cost_vector2 = [DurationCosts(i) for i in range(10000)]
    cost_vector2_sum: DurationCosts = reduce(add, cost_vector2)
    elapsed = perf_counter() - tic
    print(f"result is {cost_vector2_sum}, elapsed in {elapsed}")

    # test some representations
    print(type(cost_vector2_sum))
    print(type(DurationCosts(3)))
    print(DurationCosts(2))
    print(Decimal(2))
    print(VehicleCosts(34))

    # test serializable
    dictionary: Dict[VehicleCosts, int]
    dictionary2: Dict[DurationCosts, int]
    dictionary = {VehicleCosts(2): 3}
    dictionary2 = {DurationCosts(2): 3}
    print(dictionary)
    print(dictionary2)
