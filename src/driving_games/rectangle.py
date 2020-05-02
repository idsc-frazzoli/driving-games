from dataclasses import dataclass
from decimal import Decimal as D
from typing import Tuple

__all__ = ["Coordinates", "Rectangle", "make_rectangle"]
Coordinates = Tuple[D, D]


@dataclass
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


def make_rectangle(center: Coordinates, sides: Tuple[D, D]) -> Rectangle:
    """ Creates rectangle given center and sides. """
    c0, c1 = center
    l0, l1 = sides
    pa = c0 - l0 / 2, c1 - l1 / 2
    pb = c0 + l0 / 2, c1 + l1 / 2
    return Rectangle(pa, pb)
