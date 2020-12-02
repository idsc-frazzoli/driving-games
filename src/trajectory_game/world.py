from dataclasses import dataclass
from typing import List, Tuple

from .transitions import TransitionPath, Curve, SplineCurve

__all__ = [
    "World"
]


@dataclass
class World:
    """ Object holding all info about the world """
    ref_path: TransitionPath[float]
    """ Reference path for planner """
    left: Curve[float]
    """ Left lane boundary """
    right: Curve[float]
    """ Right lane boundary """

    def __init__(self, ref_path: TransitionPath[float],
                 left_xy: List[Tuple[float, float]],
                 right_xy: List[Tuple[float, float]]):
        self.ref_path = ref_path

        def fit_curve(p: List[Tuple[float, float]]) -> Curve[float]:
            sn = ref_path.cartesian_to_curvilinear(p)
            s, n = list(zip(*sn))
            curve = SplineCurve(s=s, z=n, order=3)
            return curve

        self.left = fit_curve(left_xy)
        self.right = fit_curve(right_xy)

    def get_bounds_at_s(self, s: List[float]) -> List[Tuple[float, float]]:
        """ Return left and right boundaries in curvilinear coordinates at progress """
        left = self.left.value_at_s(s)
        right = self.right.value_at_s(s)
        ret = list(zip(right, left))
        return ret

    def get_heading_at_s(self, s: List[float]) -> List[float]:
        return self.ref_path.heading_at_s(s)
