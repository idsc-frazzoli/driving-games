from copy import deepcopy
from functools import lru_cache
from typing import List, Dict, Tuple, Mapping, Optional
import numpy as np
from duckietown_world import SE2Transform
from bisect import bisect_right
import geometry as geo

from world import LaneSegmentHashable
from .sequence import SampledSequence, Timestamp
from .structures import VehicleState

__all__ = [
    "Trajectory",
    "FinalPoint"
]

# FinalPoint = (x_f, y_f, increase_flag)
FinalPoint = Tuple[Optional[float], Optional[float], bool]


class Trajectory:
    """ Container for trajectory - path + velocities, steering """

    traj: SampledSequence[VehicleState]
    p_final: FinalPoint
    _cache: Mapping[Timestamp, VehicleState]
    _lane: LaneSegmentHashable = None

    def __init__(self, traj: List[VehicleState], dt_samp: Timestamp = None,
                 sampled: List[VehicleState] = None, p_final: FinalPoint = None):
        times: List[Timestamp] = [t.t for t in traj]
        self.traj = SampledSequence(timestamps=times, values=traj)
        self.p_final = p_final
        if sampled is not None:
            cache = {_.t: _ for _ in sampled}
        elif dt_samp is not None:
            cache = self.upsample(dt_samp=dt_samp)
        else:
            raise Exception("One of dt_samp, sampled needed from Trajectory!")
        if p_final is not None:
            self.trim_trajectory(cache=cache, p_final=p_final)
        self._cache = cache

    def upsample(self, dt_samp: Timestamp) -> Dict[Timestamp, VehicleState]:
        times: List[Timestamp]
        x: List[VehicleState]
        times, x = zip(*self.__iter__())
        n_steps = (times[-1] - times[0]) // dt_samp
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
            step = times[0] + n * dt_samp
            if i < len(times) and step > times[i+1]:
                i += 1
                t0, dT, s0 = update_vals(idx=i)
            dt = float(step) - t0
            vx = x[i].v + u_acc[i] * dt
            st = x[i].st + u_dst[i] * dt
            s = dt * (x[i].v + vx) / 2.0
            beta = float(i) + s / s0
            cache[step] = self.interpolate(t=step, vx=vx, st=st, beta=beta)
        return cache

    def trim_trajectory(self, cache: Dict[Timestamp, VehicleState],
                        p_final: FinalPoint):
        x_f, y_f, increase = p_final
        assert x_f is None or y_f is None, "Only one of x_f, y_f should be set!"
        if x_f is not None:
            def get_z(state: VehicleState) -> float:
                return state.x
            z_f = x_f
        else:
            def get_z(state: VehicleState) -> float:
                return state.y
            z_f = y_f
        times = list(cache.keys())
        z_samp = [get_z(x) for x in cache.values()]
        if not increase:
            z0 = z_samp[0]
            z_samp = [z0 - z for z in z_samp]
            z_f = z0 - z_f
        last = bisect_right(z_samp, z_f, lo=times.index(self.traj.timestamps[-2]))
        t_end = times[last]
        for i in range(last+1, len(times)):
            cache.pop(times[i])
        self.traj.timestamps[-1] = t_end
        self.traj.values[-1] = deepcopy(cache[t_end])

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
        return self.state_to_se2_list(self.traj.values)

    def get_path_sampled(self) -> List[SE2Transform]:
        """ Returns cartesian coordinates (SE2) of trajectory at upsampled points """
        return self.state_to_se2_list(list(self._cache.values()))

    @staticmethod
    def state_to_se2_list(states: List[VehicleState]) -> List[SE2Transform]:
        ret = [Trajectory.state_to_se2(x) for x in states]
        return ret

    @staticmethod
    @lru_cache(None)
    def state_to_se2(x: VehicleState) -> SE2Transform:
        return SE2Transform(p=np.array([x.x, x.y]), theta=x.th)

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

        if self._lane is None:
            raise NotImplementedError("Interpolate works only for SE2 interpolation!")
        times = self.get_raw_sampling_points()
        i = bisect_right(times, t) - 1
        return self.interpolate(t=t, idx=i)

    def __iter__(self):
        return self.traj.__iter__()

    def __repr__(self) -> str:
        return str({f"t={round(float(k), 2)}s": v for k, v in self.traj})
