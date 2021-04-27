import cachetools
from cachetools import cached
from functools import lru_cache
from typing import List, Dict, Tuple, Optional, FrozenSet, Iterator, Union
import numpy as np
from duckietown_world import SE2Transform
from bisect import bisect_right
from networkx import DiGraph, has_path, shortest_path

from world import LaneSegmentHashable
from .sequence import Timestamp, SampledSequence
from .structures import VehicleState
from .game_def import ActionGraph

__all__ = [
    "Trajectory",
    "TrajectoryGraph",
    "FinalPoint"
]

# FinalPoint = (x_f, y_f, increase_flag)
FinalPoint = Tuple[Optional[float], Optional[float], bool]


class Trajectory:
    """ Container for trajectory - sequence of vehicle states """
    traj: List["Trajectory"]
    states: SampledSequence[VehicleState]
    lane: LaneSegmentHashable

    def __init__(self, values: List[Union[VehicleState, "Trajectory"]],
                 lane: LaneSegmentHashable,
                 p_final: Optional[FinalPoint] = None,
                 states: Optional[Tuple[VehicleState, VehicleState]] = None):
        assert len(values) > 0
        self.lane = lane
        if all(isinstance(val, Trajectory) for val in values):
            self.traj = values
            x_t: Dict[Timestamp, VehicleState] = {t: x for val in values for t, x in val}
            self.states = SampledSequence(timestamps=list(x_t.keys()), values=list(x_t.values()))
        elif all(isinstance(val, VehicleState) for val in values):
            if states is not None:
                values[0] = states[0]
                values[-1] = states[-1]
            if p_final is not None:
                self._trim_trajectory(states=values, p_final=p_final)
            times: List[Timestamp] = [x.t for x in values]
            self.states = SampledSequence(timestamps=times, values=values)
            self.traj = []
        else:
            raise TypeError(f"Input is of wrong type - {type(values[0])}!")

    @staticmethod
    @cached(cache={}, key=lambda states, lane, values, p_final: cachetools.keys.hashkey((states, lane)))
    def create(states: Tuple[VehicleState, VehicleState], lane: LaneSegmentHashable,
               values: List[VehicleState], p_final: FinalPoint = None):
        return Trajectory(values=values, lane=lane, p_final=p_final, states=states)

    @staticmethod
    def state_to_se2_list(states: List[VehicleState]) -> List[SE2Transform]:
        ret = [Trajectory.state_to_se2(x) for x in states]
        return ret

    @staticmethod
    @lru_cache(None)
    def state_to_se2(x: VehicleState) -> SE2Transform:
        return SE2Transform(p=np.array([x.x, x.y]), theta=x.th)

    @staticmethod
    def _trim_trajectory(states: List[VehicleState], p_final: FinalPoint):
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
        times = [x.t for x in states]
        z_samp = [get_z(x) for x in states]
        if not increase:
            z0 = z_samp[0]
            z_samp = [z0 - z for z in z_samp]
            z_f = z0 - z_f
        if z_f < z_samp[0] or z_f > z_samp[-1]:
            return states
        last = bisect_right(z_samp, z_f)
        for _ in range(last + 1, len(times)):
            states.pop()

    def __iter__(self) -> Iterator[Tuple[Timestamp, VehicleState]]:
        return self.states.__iter__()

    def __len__(self):
        return len(self.states)

    def get_lane(self) -> LaneSegmentHashable:
        return self.lane

    def get_trajectories(self) -> List["Trajectory"]:
        if len(self.traj) == 0:
            return [self]
        return self.traj

    def get_sampling_points(self) -> List[Timestamp]:
        """ Returns timestamps of trajectory points """
        return self.states.get_sampling_points()

    def get_path_sampled(self) -> List[SE2Transform]:
        """ Returns cartesian coordinates (SE2) of transition states """
        return self.state_to_se2_list(self.states.values)

    def get_start(self) -> Timestamp:
        return self.states.get_start()

    def get_end(self) -> Timestamp:
        return self.states.get_end()

    def at(self, t: Timestamp) -> VehicleState:
        return self.states.get_interp(t)

    def __repr__(self) -> str:
        states: Dict[str, VehicleState] = {}

        def add_entry(t_stamp):
            states[f"t={round(float(t_stamp), 2)}s"] = self.at(t_stamp)
        t = self.get_start()
        while t <= self.get_end():
            add_entry(t_stamp=t)
            t += Timestamp("1")
        add_entry(t_stamp=self.get_end())
        return str(states)

    def __add__(self, other: Optional["Trajectory"]) -> "Trajectory":
        if other is None:
            return self
        x1, x2 = self.at(self.get_end()), other.at(other.get_start())
        if not x1.is_close(x2):
            raise ValueError(f"Transitions not continuous - {x1, x2}")
        return Trajectory(values=self.traj+other.traj)

    def starts_with(self, start: "Trajectory") -> bool:
        if len(start) > len(self): return False
        for t in start.get_sampling_points():
            if start.at(t) != self.at(t): return False
        return True


class TrajectoryGraph(ActionGraph[Trajectory], DiGraph):
    """ Structure for storing all trajectories """
    origin: VehicleState
    lane: LaneSegmentHashable
    trajectories: Dict[Tuple[VehicleState, VehicleState], Trajectory]

    def __init__(self, origin: VehicleState, lane: LaneSegmentHashable, **attr):
        super().__init__(**attr)
        self.origin = origin
        self.lane = lane
        self.trajectories = {}

    def add_node(self, state: VehicleState, **attr):
        super(TrajectoryGraph, self).add_node(node_for_adding=state, **attr)

    def check_node(self, node: VehicleState):
        if node not in self.nodes:
            raise ValueError(f"{node} not in graph!")

    def add_edge(self, trajectory: Trajectory, **attr):
        source, target = trajectory.at(trajectory.get_start()), trajectory.at(trajectory.get_end())
        attr["transition"] = trajectory
        if target not in self.nodes:
            self.add_node(state=target, gen=self.nodes[source]["gen"] + 1)

        super(TrajectoryGraph, self).add_edge(u_of_edge=source, v_of_edge=target, **attr)
        self.trajectories[(source, target)] = trajectory

    def get_all_trajectories(self, source: VehicleState) -> FrozenSet[Trajectory]:
        if source not in self.nodes:
            raise ValueError(f"Source node ({source}) not in graph!")

        successors = [self.get_trajectory_edge(source=source, target=target)
                      for target in self.successors(source)]
        return frozenset(successors)

    def get_trajectory_edge(self, source: VehicleState, target: VehicleState) -> Trajectory:
        states = (source, target)
        if states not in self.trajectories:
            raise ValueError(f"{states} not found in transitions!")
        return self.trajectories[(source, target)]

    @lru_cache(None)
    def get_trajectory(self, source: VehicleState, target: VehicleState) -> Trajectory:
        self.check_node(source)
        self.check_node(target)
        if not has_path(G=self, source=source, target=target):
            raise ValueError(f"No path exists between {source, target}!")

        nodes = shortest_path(G=self, source=source, target=target)
        traj: List[Trajectory] = []
        for node1, node2 in zip(nodes[:-1], nodes[1:]):
            traj.append(self.get_trajectory_edge(source=node1, target=node2))
        return Trajectory(values=traj, lane=self.lane)
