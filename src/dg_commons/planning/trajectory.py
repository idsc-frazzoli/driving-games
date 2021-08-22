from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from duckietown_world import SE2Transform

from dg_commons.sequence import DgSampledSequence, X

__all__ = [
    "Trajectory",
    # "TrajectoryGraph",
    "FinalPoint"
]

# FinalPoint = (x_f, y_f, increase_flag)
FinalPoint = Tuple[Optional[float], Optional[float], bool]


@dataclass
class Trajectory(DgSampledSequence[X]):
    """ Container for a trajectory as a sampled sequence """

    lane: Optional[DgLanelet] = None
    """ If thee trajectory has to be referenced wrt to a lane"""

    def trim_trajectory(self, p_final: FinalPoint) -> bool:
        """ Trims trajectory till p_final (if longer) and returns if trimming was performed or not """
        # todo
        # x_f, y_f, increase = p_final
        # assert x_f is None or y_f is None, "Only one of x_f, y_f should be set!"
        # if x_f is not None:
        #     def get_z(state: VehicleState) -> float:
        #         return state.x
        #
        #     z_f = x_f
        # else:
        #     def get_z(state: VehicleState) -> float:
        #         return state.y
        #
        #     z_f = y_f
        # times = [x.t for x in states]
        # z_samp = [get_z(x) for x in states]
        # if not increase:
        #     z0 = z_samp[0]
        #     z_samp = [z0 - z for z in z_samp]
        #     z_f = z0 - z_f
        # if z_f < z_samp[0] or z_f > z_samp[-1]:
        #     return False
        # last = bisect_right(z_samp, z_f)
        # for _ in range(last + 1, len(times)):
        #     states.pop()
        return True

    def get_path(self) -> List[SE2Transform]:
        """ Returns cartesian coordinates (SE2) of transition states """
        return [SE2Transform(p=np.array([x.x, x.y]), theta=x.th) for x in self.values]

    def __add__(self, other: Optional["Trajectory"]) -> "Trajectory":
        # todo
        pass
