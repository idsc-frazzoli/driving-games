from typing import List, Dict, Tuple, Mapping
import numpy as np
from duckietown_world import SE2Transform
from bisect import bisect_right
import geometry as geo

from world import LaneSegmentHashable
from .sequence import SampledSequence, Timestamp
from .structures import VehicleState

__all__ = [
    "Trajectory",
]


class Trajectory:
    """ Container for trajectory - path + velocities, steering """

    traj: SampledSequence[VehicleState]
    dt_samp: Timestamp
    _cache: Mapping[Timestamp, VehicleState]
    _lane: LaneSegmentHashable

    def __init__(self, traj: List[VehicleState], dt_samp: Timestamp):
        times: List[Timestamp] = [t.t for t in traj]
        self.traj = SampledSequence(timestamps=times, values=traj)
        self.dt_samp = dt_samp
        self.upsample()

    def upsample(self):
        times: List[Timestamp]
        x: List[VehicleState]
        times, x = zip(*self.__iter__())
        n_steps = (times[-1] - times[0]) // self.dt_samp
        u_acc: List[float] = []
        u_dst: List[float] = []
        for i in range(len(times)-1):
            x0, x1 = x[i], x[i+1]
            dT = float(times[i+1] - times[i])
            u_acc.append((x1.v - x0.v) / dT)
            u_dst.append((x1.st - x0.st) / dT)

        def update_vals(idx: int) -> Tuple[float, float, float]:
            t01 = float(times[idx])
            dT1 = float(times[idx + 1]) - t01
            s01 = (x[idx].v + u_acc[idx] * dT1 / 2.0) * dT1
            return t01, dT1, s01

        i = 0
        t0, dT, s0 = update_vals(idx=i)
        self._lane = LaneSegmentHashable(width=0.1, control_points=self.get_path())
        cache: Dict[Timestamp, VehicleState] = {}

        for n in np.arange(0, n_steps + 1, dtype=int):
            step = times[0] + n * self.dt_samp
            if i < len(times) and step > times[i+1]:
                i += 1
                t0, dT, s0 = update_vals(idx=i)
            dt = float(step) - t0
            vx = x[i].v + u_acc[i] * dt
            st = x[i].st + u_dst[i] * dt
            s = dt * (x[i].v + vx) / 2.0
            beta = float(i) + s / s0
            cache[step] = self.interpolate(t=step, vx=vx, st=st, beta=beta)
        self._cache = cache

    def get_sequence(self) -> SampledSequence[VehicleState]:
        """ Returns sequence of trajectory points """
        return self.traj

    def get_raw_sampling_points(self) -> List[Timestamp]:
        """ Returns timestamps of actual trajectory points """
        return self.traj.get_sampling_points()

    def get_sampling_points(self) -> List[Timestamp]:
        """ Returns timestamps of upsampled trajectory points """
        return list(self._cache.keys())

    def get_sampled_trajectory(self):
        return self._cache.items().__iter__()

    def get_path(self) -> List[SE2Transform]:
        """ Returns cartesian coordinates (SE2) of trajectory """
        return self.state_to_se2(self.traj.values)

    def get_path_sampled(self) -> List[SE2Transform]:
        """ Returns cartesian coordinates (SE2) of trajectory at upsampled points """
        return self.state_to_se2(list(self._cache.values()))

    @staticmethod
    def state_to_se2(states: List[VehicleState]) -> List[SE2Transform]:
        ret = [SE2Transform(p=np.array([x.x, x.y]), theta=x.th) for x in states]
        return ret

    def interpolate(self, t: Timestamp, idx: int = None,
                    vx: float = None, st: float = None,
                    beta: float = None) -> VehicleState:
        assert idx is not None or not (beta is None or vx is None or st is None), \
            "Incorrect inputs for interpolate"
        if beta is None:
            times, x = zip(*self.__iter__())
            t0 = float(times[idx])
            dT = float(times[idx + 1]) - t0
            u_acc = (x[idx+1].v - x[idx].v) / dT
            u_dst = (x[idx+1].st - x[idx].st) / dT
            s0 = (x[idx].v + u_acc * dT / 2.0) * dT
            dt = float(t) - t0
            vx = x[idx].v + u_acc * dt
            st = x[idx].st + u_dst * dt
            s = dt * (x[idx].v + vx) / 2.0
            beta = float(idx) + s / s0
        q = self._lane.center_point(beta=beta)
        pos, ang, _ = geo.translation_angle_scale_from_E2(pose=q)
        return VehicleState(x=pos[0], y=pos[1], th=ang, v=vx, st=st, t=t)

    def at(self, t: Timestamp) -> VehicleState:
        """Returns value at requested timestamp,
        Interpolates between timestamps, holds at the extremes"""
        if t in self._cache.keys():
            return self._cache[t]
        if t < self.traj.get_start():
            return self.traj.at(self.traj.get_start())
        elif t > self.traj.get_end():
            return self.traj.at(self.traj.get_end())

        times = self.get_raw_sampling_points()
        i = bisect_right(times, t)
        return self.interpolate(t=t, idx=i)

    def __iter__(self):
        return self.traj.__iter__()

    def __repr__(self) -> str:
        return str({f"t={round(float(k), 2)}s": v for k, v in self.traj})
