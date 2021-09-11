from dataclasses import dataclass, replace
from functools import partial
from typing import List, Optional

import numpy as np
from duckietown_world import SE2Transform

from dg_commons.sequence import DgSampledSequence
from sim.models.vehicle import VehicleState

__all__ = [
    "Trajectory",
]


@dataclass
class Trajectory(DgSampledSequence[VehicleState]):
    """ Container for a trajectory as a sampled sequence """

    def as_path(self) -> List[SE2Transform]:
        """ Returns cartesian coordinates (SE2) of transition states """
        return [SE2Transform(p=np.array([x.x, x.y]), theta=x.theta) for x in self.values]

    def apply_SE2transform(self, transform: SE2Transform):
        def _applySE2(x: VehicleState, t: SE2Transform) -> VehicleState:
            return replace(x, x=x.x + t.p[0], y=x.y + t.p[1], theta=x.theta + t.theta)

        f = partial(_applySE2, t=transform)
        return self.transform_values(f=f, YT=VehicleState)

    def is_connectable(self, other: 'Trajectory', tol=1e-3) -> bool:
        """
        Any primitive whose initial state's velocity and steering angle are equal to those of the current primitive is
        deemed connectable.

        :param other: the motion primitive to which the connectivity is examined
        """
        diff = self.at(self.get_end()) - other.at(other.get_end())
        return abs(diff.vx) < tol and abs(diff.delta) < tol

    def __add__(self, other: Optional["Trajectory"]) -> "Trajectory":
        assert self.is_connectable(other)
        # todo
        pass
