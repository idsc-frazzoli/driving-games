from copy import deepcopy
from cachetools import cached
from cachetools.keys import hashkey
from functools import lru_cache
from typing import List, Dict, Tuple, Mapping, Optional
import numpy as np
from duckietown_world import SE2Transform
from bisect import bisect_right
from networkx import DiGraph, has_path, shortest_path

from .sequence import Timestamp
from .structures import VehicleState

__all__ = [
    "Transition",
    "Trajectory",
    "TransitionGraph",
    "FinalPoint"
]

# FinalPoint = (x_f, y_f, increase_flag)
FinalPoint = Tuple[Optional[float], Optional[float], bool]


class Transition:
    """ Container for individual transitions that make up trajectory"""

    states: Tuple[VehicleState, VehicleState]
    _cache: Mapping[Timestamp, VehicleState]

    def __init__(self, states: Tuple[VehicleState, VehicleState],
                 sampled: List[VehicleState], p_final: FinalPoint):
        self.states = states
        cache = {_.t: _ for _ in sampled}
        if p_final is not None:
            self._trim_transition(cache=cache, p_final=p_final)
        self._cache = cache

    @staticmethod
    @cached(cache={}, key=lambda states, sampled, p_final: hashkey(states))
    def create(states: Tuple[VehicleState, VehicleState],
               sampled: List[VehicleState] = None, p_final: FinalPoint = None):
        return Transition(states=states, sampled=sampled, p_final=p_final)

    def _trim_transition(self, cache: Dict[Timestamp, VehicleState],
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
        if z_f < z_samp[0] or z_f > z_samp[-1]:
            return
        last = bisect_right(z_samp, z_f)
        t_end = times[last]
        for i in range(last + 1, len(times)):
            cache.pop(times[i])
        start, _ = self.states
        finish = deepcopy(cache[t_end])
        self.states = (start, finish)

    def get_raw_sampling_points(self) -> List[Timestamp]:
        """ Returns timestamps of actual transition points """
        return [t for t, x in self]

    def get_sampling_points(self) -> List[Timestamp]:
        """ Returns timestamps of upsampled trajectory points """
        return list(self._cache.keys())

    def get_sampled_trajectory(self):
        return self._cache.items().__iter__()

    def get_path(self) -> List[SE2Transform]:
        """ Returns cartesian coordinates (SE2) of transition """
        return self.state_to_se2_list([x for x in self.states])

    def get_path_sampled(self) -> List[SE2Transform]:
        """ Returns cartesian coordinates (SE2) of transition at upsampled points """
        return self.state_to_se2_list(list(self._cache.values()))

    @staticmethod
    def state_to_se2_list(states: List[VehicleState]) -> List[SE2Transform]:
        ret = [Transition.state_to_se2(x) for x in states]
        return ret

    @staticmethod
    @lru_cache(None)
    def state_to_se2(x: VehicleState) -> SE2Transform:
        return SE2Transform(p=np.array([x.x, x.y]), theta=x.th)

    def at(self, t: Timestamp) -> VehicleState:
        """ Returns value at requested timestamp, Interpolates between timestamps """
        if t in self._cache.keys():
            return self._cache[t]
        start, finish = self.states
        if t < start.t or t > finish.t:
            raise ValueError(f"{t} doesn't lie within {start.t, finish.t}!")

        raise NotImplementedError("Interpolate not implemented!")

    def __iter__(self):
        for state in self.states:
            yield state.t, state

    def __repr__(self) -> str:
        return str({f"t={round(float(k), 2)}s": v for k, v in self})


class Trajectory:
    """ Container for trajectory - sequence of transitions """
    traj: List[Transition]
    times: List[Timestamp]

    def __init__(self, traj: List[Transition]):
        assert len(traj) > 0
        self.traj = traj
        start, _ = traj[0].states
        self.times = [start.t] + [tran.states[1].t for tran in traj]

    def __iter__(self):
        return self.traj.__iter__()

    def __len__(self):
        return len(self.traj)

    def __getitem__(self, item: int):
        if item < 0 or item > len(self):
            raise ValueError(f"Index {item} out of range - {0, len(self)}")
        return self.traj[item]

    @lru_cache(None)
    def get_sampled_trajectory(self) -> Tuple[List[Timestamp], List[VehicleState]]:
        times: List[Timestamp] = [Timestamp("0")]
        states: List[VehicleState] = [VehicleState.zero()]
        for trans in self.traj:
            t, x = map(list, zip(*trans.get_sampled_trajectory()))
            times = times[:-1] + t
            states = states[:-1] + x
        return times, states

    def get_end(self) -> Timestamp:
        return self.times[-1]

    def at(self, t: Timestamp) -> VehicleState:
        if t <= self.times[0]:
            start, _ = self.traj[0].states
            return start
        if t >= self.times[-1]:
            _, finish = self.traj[-1].states
            return finish
        i = bisect_right(self.times, t) - 1
        return self.traj[i].at(t=t)

    def __repr__(self) -> str:
        return str({f"t={round(float(k), 2)}s": v for trans in self.traj for k, v in trans})


class TransitionGraph(DiGraph):
    """ Structure for storing all trajectories """
    origin: VehicleState
    transitions: Dict[Tuple[VehicleState, VehicleState], Transition]

    def __init__(self, origin: VehicleState, **attr):
        super().__init__(**attr)
        self.origin = origin
        self.transitions = {}

    def add_node(self, state: VehicleState, **attr):
        super(TransitionGraph, self).add_node(node_for_adding=state, **attr)

    def check_node(self, node: VehicleState):
        if node not in self.nodes:
            raise ValueError(f"{node} not in graph!")

    def add_edge(self, transition: Transition, **attr):
        source, target = transition.states
        attr["transition"] = transition
        if target not in self.nodes:
            self.add_node(state=target, gen=self.nodes[source]["gen"] + 1)

        super(TransitionGraph, self).add_edge(u_of_edge=source, v_of_edge=target, **attr)
        self.transitions[transition.states] = transition

    def get_transition(self, source: VehicleState, target: VehicleState) -> Transition:
        states = (source, target)
        if states not in self.transitions:
            raise ValueError(f"{states} not found in transitions!")
        return self.transitions[(source, target)]

    @lru_cache(None)
    def get_trajectory(self, source: VehicleState, target: VehicleState) -> Trajectory:
        self.check_node(source)
        self.check_node(target)
        if not has_path(G=self, source=source, target=target):
            raise ValueError(f"No path exists between {source, target}!")

        nodes = shortest_path(G=self, source=source, target=target)
        traj: List[Transition] = []
        for node1, node2 in zip(nodes[:-1], nodes[1:]):
            traj.append(self.get_transition(source=node1, target=node2))
        return Trajectory(traj=traj)
