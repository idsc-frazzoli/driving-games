from dataclasses import dataclass
from decimal import Decimal as D
from itertools import product
from typing import List, Tuple

__all__ = ["Coordinates", "Rectangle", "make_rectangle"]
Coordinates = Tuple[D, D]


@dataclass(frozen=True)
class Rectangle:
    """ Represents a rectangle """

    bottom_left: Coordinates
    top_right: Coordinates

    def sizes(self) -> Tuple[D, D]:
        return (
            self.top_right[0] - self.bottom_left[0],
            self.top_right[1] - self.bottom_left[1],
        )

    def area(self) -> D:
        a, b = self.sizes()
        return a * b

    def contains(self, c: Coordinates) -> bool:
        bl = self.bottom_left
        tr = self.top_right
        return (bl[0] <= c[0] <= tr[0]) and (bl[1] <= c[1] <= tr[1])


def get_rectangle_points_around(r: Rectangle) -> List[Coordinates]:
    bl = r.bottom_left
    tr = r.top_right

    n = 6
    res = []
    for i, j in product(range(n), range(n)):
        alpha = D(i) / D(n - 1)
        beta = D(j) / D(n - 1)
        x = bl[0] * alpha + (1 - alpha) * tr[0]
        y = bl[1] * beta + (1 - beta) * tr[1]
        res.append((D(x), D(y)))
    return res


def get_rectangle_countour(r: Rectangle) -> List[Tuple[float, float]]:
    bl = list(map(float, r.bottom_left))
    tr = list(map(float, r.top_right))
    return [(bl[0], bl[1]), (tr[0], bl[1]), (tr[0], tr[1]), (bl[0], tr[1]), (bl[0], bl[1])]


def make_rectangle(center: Coordinates, sides: Tuple[D, D]) -> Rectangle:
    """ Creates rectangle given center and sides. """
    c0, c1 = center
    l0, l1 = sides
    pa = c0 - l0 / 2, c1 - l1 / 2
    pb = c0 + l0 / 2, c1 + l1 / 2
    return Rectangle(pa, pb)
