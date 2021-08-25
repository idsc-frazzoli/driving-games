from functools import lru_cache
from typing import List, Dict, Tuple, Optional, FrozenSet, Iterator, Union

import cachetools
import numpy as np
from cachetools import cached
from duckietown_world import SE2Transform
from networkx import DiGraph, has_path, shortest_path
from shapely.geometry import Polygon, Point

from dg_commons.planning.lanes import DgLanelet
from dg_commons.sequence import Timestamp, DgSampledSequence
from .game_def import ActionGraph
from .structures import VehicleState

__all__ = [
    "Trajectory",
    "TrajectoryGraph",
]


class Trajectory:
    """ Container for a trajectory - sequence of vehicle states """

    traj: List["Trajectory"]
    """ A trajectory can also be made up of multiple smaller trajectories.
        This is used for evaluation of trajectory metrics where the 
        outcomes are cached using the trajectory as the key """

    states: DgSampledSequence[VehicleState]
    """ The upsampled sequence of vehicle states """
    lane: DgLanelet
    """ The reference lane used to generate the trajectory """
    goal: Optional[Polygon]
    """ The goal region for the trajectory """

    def __init__(self, values: List[Union[VehicleState, "Trajectory"]],
                 lane: DgLanelet,
                 goal: Optional[Polygon] = None,
                 states: Optional[Tuple[VehicleState, VehicleState]] = None):
        assert len(values) > 0
        self.lane = lane
        self.goal = goal
        if all(isinstance(val, Trajectory) for val in values):
            self.traj = values
            x_t: Dict[Timestamp, VehicleState] = {t: x for val in values for t, x in val}
            self.states = DgSampledSequence(timestamps=list(x_t.keys()), values=list(x_t.values()))
        elif all(isinstance(val, VehicleState) for val in values):
            if states is not None:
                values[0] = states[0]
                values[-1] = states[-1]
            self.trim_trajectory(states=values, goal=goal)
            times: List[Timestamp] = [x.t for x in values]
            self.states = DgSampledSequence(timestamps=times, values=values)
            self.traj = []
        else:
            raise TypeError(f"Input is of wrong type - {type(values[0])}!")

    @staticmethod
    @cached(cache={}, key=lambda states, lane, values, goal: cachetools.keys.hashkey((states, lane)))
    def create(states: Tuple[VehicleState, VehicleState], lane: DgLanelet,
               values: List[VehicleState], goal: Optional[Polygon] = None):
        return Trajectory(values=values, lane=lane, goal=goal, states=states)

    @staticmethod
    def state_to_se2_list(states: List[VehicleState]) -> List[SE2Transform]:
        ret = [Trajectory.state_to_se2(x) for x in states]
        return ret

    @staticmethod
    @lru_cache(None)
    def state_to_se2(x: VehicleState) -> SE2Transform:
        return SE2Transform(p=np.array([x.x, x.y]), theta=x.th)

    @staticmethod
    def trim_trajectory(states: List[VehicleState], goal: Optional[Polygon]) -> bool:
        """ Trims trajectory till goal region (if longer) and returns if trimming was performed or not """
        if goal is None:
            return False
        goal_idx = Trajectory.get_in_goal_index(states=states, goal=goal)
        if goal_idx is None:
            return False
        n_states = len(states)
        for _ in range(goal_idx + 1, n_states):
            states.pop()
        return True

    @staticmethod
    def get_in_goal_index(states: List[VehicleState], goal: Polygon) -> Optional[int]:
        in_goal = [goal.contains(Point(x.x, x.y)) for x in states]
        try:
            last = in_goal.index(True)
            return last
        except:
            return None

    def __iter__(self) -> Iterator[Tuple[Timestamp, VehicleState]]:
        return self.states.__iter__()

    def __len__(self):
        return len(self.states)

    def get_lane(self) -> Tuple[DgLanelet, Optional[Polygon]]:
        return self.lane, self.goal

    def get_trajectories(self) -> List["Trajectory"]:
        if len(self.traj) == 0:
            return [self]
        return self.traj

    def get_sampling_points(self) -> List[Timestamp]:
        """ Returns timestamps of trajectory points """
        return self.states.get_sampling_points()

    def get_path_sampled(self) -> List[SE2Transform]:
        """ Returns cartesian coordinates (SE2) of transition states """
        return self.state_to_se2_list(self.states._values)

    def get_start(self) -> Timestamp:
        return self.states.get_start()

    def get_end(self) -> Timestamp:
        return self.states.get_end()

    def at(self, t: Timestamp) -> VehicleState:
        return self.states.at_interp(t)

    def __repr__(self) -> str:
        states: Dict[str, VehicleState] = {}

        def add_entry(t_stamp):
            states[f"t={round(float(t_stamp), 2)}s"] = self.at(t_stamp)

        t = self.get_start()
        while t <= self.get_end():
            add_entry(t_stamp=t)
            t += 1
        add_entry(t_stamp=self.get_end())
        return str(states)

    def __add__(self, other: Optional["Trajectory"]) -> "Trajectory":
        """ Combines trajectories into a bigger trajectory """
        if other is None:
            return self
        x1, x2 = self.at(self.get_end()), other.at(other.get_start())
        if not x1.is_close(x2):
            raise ValueError(f"Transitions not continuous - {x1, x2}")
        lane, goal = other.get_lane()
        return Trajectory(values=self.traj + other.traj, lane=lane, goal=goal)

    def starts_with(self, start: "Trajectory") -> bool:
        if len(start) > len(self): return False
        for t in start.get_sampling_points():
            if start.at(t) != self.at(t): return False
        return True


class TrajectoryGraph(ActionGraph[Trajectory], DiGraph):
    """ Structure for storing a graph of trajectory states """
    origin: VehicleState
    """ Origin of the graph of states """
    lane: DgLanelet
    """ Reference lane used to generate trajectories """
    goal: Optional[Polygon]
    """ Goal region for the trajectories """
    trajectories: Dict[Tuple[VehicleState, VehicleState], Trajectory]
    """ Store trajectories based on terminal states """

    def __init__(self, origin: VehicleState, lane: DgLanelet, goal: Optional[Polygon] = None, **attr):
        super().__init__(**attr)
        self.origin = origin
        self.lane = lane
        self.goal = goal
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
        return Trajectory(values=traj, lane=self.lane, goal=self.goal)
