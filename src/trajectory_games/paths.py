from typing import List
import numpy as np
from duckietown_world import SE2Transform

from .sequence import SampledSequence, Timestamp
from .structures import VehicleState

__all__ = [
    "Trajectory",
]


class Trajectory:
    """ Container for trajectory - path + velocities, steering """

    traj: SampledSequence[VehicleState]

    def __init__(self, traj: List[VehicleState]):
        times: List[Timestamp] = [t.t for t in traj]
        self.traj = SampledSequence(timestamps=times, values=traj)

    def get_sequence(self) -> SampledSequence[VehicleState]:
        """ Returns sequence of trajectory points """
        return self.traj

    def get_sampling_points(self) -> List[Timestamp]:
        """ Returns timestamps of trajectory points """
        return self.traj.get_sampling_points()

    def get_path(self) -> List[SE2Transform]:
        """ Returns cartesian coordinates (SE2) of trajectory """
        ret = [SE2Transform(p=np.array([x.x, x.y]), theta=x.th) for _, x in self.traj]
        return ret

    def at(self, t: Timestamp) -> VehicleState:
        return self.traj.get_interp(t)

    def __iter__(self):
        return self.traj.__iter__()

    def __repr__(self) -> str:
        return str({f"t={round(float(k), 2)}s": v for k, v in self.traj})
