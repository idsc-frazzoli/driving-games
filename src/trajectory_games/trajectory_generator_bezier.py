import dataclasses

import numpy as np
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass
import random
from shapely.geometry import Point

from dg_commons import Timestamp
from dg_commons.dynamics import BicycleDynamics
from dg_commons.sim.models.vehicle import VehicleState, VehicleCommands
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from math import pi

from dg_commons.sim.models.vehicle_utils import VehicleParameters


@dataclass
class BezierCurveGeneratorParams:
    """Class for storing all relevant parameters for generating Bezier curves"""

    n_curves: int  # number of curves to generate
    goal_sampling_rectangle: Dict[
        str, float
    ]  # [length, width] is a rectangle centered around goal in which to sample new goals
    n_points: int  # number of control points to add as deviations from straight line from start to goal
    max_deviations: float  # maximum deviation of the control points from central line
    deviations_distr: str  # which distribution to use to sample deviation distances: uniform, ...
    deviation_locations_distr: str  # distribution to use to calculate the progress along which new points are added #todo define this (if needed and if a distr)
    min_radius: float  # min radius of curvature

    @staticmethod
    def default():
        return BezierCurveGeneratorParams(
            n_curves=40,
            goal_sampling_rectangle={"x": 2.0, "y": 3.0},
            n_points=2,
            deviation_locations_distr="uniform",
            max_deviations=10.0,
            deviations_distr="uniform",
            min_radius=0.5,
        )


@dataclass
class TrajectoryGeneratorParams(BezierCurveGeneratorParams):
    sampling_time: Timestamp
    n_states: int
    min_vel: float  # todo get those two from bicycle model
    max_vel: float
    min_acc: float
    max_acc: float
    min_ddelta: float
    max_ddelta: float

    @classmethod
    def default(cls):
        return TrajectoryGeneratorParams(
            **dataclasses.asdict(super().default()),
            sampling_time=1.0,
            n_states=10,
            min_vel=0.0,
            max_vel=10.0,
            min_acc=-10.0,
            max_acc=10.0,
            min_ddelta=-2.0,  # todo these are probably too large?
            max_ddelta=2.0
        )


class BezierCurve:
    def __init__(self, control_points: List[Point]):
        self.control_points = control_points
        self.points: List[Point] = []

    def compute_bezier(self, t: float, control_points: Optional[List[Point]] = None):
        """Recursive definition of Bézier curves of degree n
        :param t: value in range [0,1] for which to compute coordinate on Bézier curves
        :param control_points: points that define Bézier curve. Defined in init method. Here
        accepted as inputs only for internal recursive call."""

        if not control_points:
            control_points = self.control_points
        if 0 <= t <= 1:
            idx = 0

            # zeroth order compute_bezier curve
            if len(control_points) == 1:
                return np.array(control_points[0].coords[:][0])

            # recursive formulation
            else:
                idx = idx + 1
                return (1 - t) * self.compute_bezier(t, control_points[:-idx]) + t * self.compute_bezier(
                    t, control_points[idx:]
                )
        else:
            assert False, "t should be in range [0,1]"

    # todo if not precise enough use second order derivative of bezier curve
    def curvature(self, t: float) -> float:
        """Numerical approximation of curvature for a bezier curve"""
        t_before = t * 0.99
        t_after = t * 1.01
        if t == 0.0:
            t_before = 0.0
            t_after = 0.02
        if t_after > 1.0:
            t_before = 0.98
            t_after = 1.0
        tangent_before = self.tangent(t=t_before)
        tangent_after = self.tangent(t=t_after)
        if np.array_equal(tangent_before, tangent_after):  # Infinite curvature
            return 0.0
        theta = np.arccos(tangent_before.dot(tangent_after))
        return abs(theta / (t_after - t_before))

    def radius(self, t: float) -> float:
        curvature = self.curvature(t=t)
        if curvature == 0.0:
            return 99999999999.0  # Infinite radius of curvature
        return 1 / abs(curvature)

    def max_curvature(self):
        ts = np.linspace(0, 1, num=50)
        curvatures = [self.curvature(t=t) for t in ts]
        return max(curvatures)

    def min_radius(self):
        ts = np.linspace(0, 1, num=50)
        radiuses = [self.radius(t=t) for t in ts]
        return min(radiuses)

    def tangent(self, t: float) -> np.ndarray:
        """Numerical approximation of tangent vector for a bezier curve"""
        t_before = t * 0.98
        t_after = t * 1.02
        if t == 0.0:
            t_before = 0.0
            t_after = 0.002
        if t_after > 1.0:
            t_before = 0.998
            t_after = 1.0
        point_before = self.compute_bezier(t=t_before)
        point_before = point_before
        point_after = self.compute_bezier(t=t_after)
        point_after = point_after
        tangent = point_after - point_before
        return tangent / np.linalg.norm(tangent)

    # todo make robust
    def theta(self, t: float):
        tangent = self.tangent(t=t)
        return np.arctan2(tangent[0], tangent[1])

    def get_control_points(self):
        return self.control_points


# todo: we could resample curves around each curve by keeping start, goals, but resampling control points
class CurveGenerator:
    """Class to generate a set of (Bezier) curves starting from the current state of an agent"""

    def __init__(
        self, start: Point, goal: Point, params: Optional[BezierCurveGeneratorParams] = None
    ):  # todo extend this to region if needed
        self.start = start
        self.goal = goal
        if params:
            self.params = params
        else:
            self.params: BezierCurveGeneratorParams = BezierCurveGeneratorParams.default()
        self.new_goals = self._sample_other_goals()

    def _sample_other_goals(self):
        new_xs = [
            random.random() * self.params.goal_sampling_rectangle["x"] + self.goal.x
            for _ in range(self.params.n_curves)
        ]
        new_ys = [
            random.random() * self.params.goal_sampling_rectangle["y"] + self.goal.y
            for _ in range(self.params.n_curves)
        ]
        new_goals: List[Point] = [Point(new_xs[i], new_ys[i]) for i in range(self.params.n_curves)]
        return new_goals

    def get_curve_bundle(self) -> List[BezierCurve]:
        assert len(self.new_goals) == self.params.n_curves
        first_curve = BezierCurve(control_points=self.sample_control_points((self.start, self.goal)))
        if first_curve.min_radius() > self.params.min_radius:
            bezier_curves = [first_curve]
        else:
            bezier_curves = []
        for goal in self.new_goals:
            curve = BezierCurve(control_points=self.sample_control_points((self.start, goal)))
            if curve.min_radius() > self.params.min_radius:
                bezier_curves.append(curve)
        return bezier_curves

    # sample deviation points from straight line
    def sample_control_points(self, line: Tuple[Point, Point]) -> List[Point]:
        x_diff = line[1].coords[:][0][0] - line[0].coords[:][0][0]
        y_diff = line[1].coords[:][0][1] - line[0].coords[:][0][1]
        tangent = (x_diff, y_diff)
        length = np.linalg.norm(tangent)
        tangent = tangent / length
        normal = np.array((-1.0 * tangent[1], tangent[0]))  # todo check types are equal between normal and tangent
        n_points = self.params.n_points
        if self.params.deviation_locations_distr == "uniform":
            deviation_locations = [random.random() * abs(length) for _ in range(n_points)]
        else:
            deviation_locations = []
            assert NotImplementedError

        if self.params.deviations_distr == "uniform":
            deviations = [(random.random() - 0.5) * abs(self.params.max_deviations) for _ in range(n_points)]
        else:
            deviations = []
            assert NotImplementedError

        control_points = [
            line[0].coords[:][0] + tangent * deviation_locations[i] + normal * deviations[i] for i in range(n_points)
        ]
        control_points.insert(0, line[0])
        control_points.append(line[1])
        control_points = [Point(point) for point in control_points]
        return control_points


class TrajectoryGenerator:
    # todo handle sampling time
    def __init__(
        self,
        init_state: VehicleState,
        goal_state: VehicleState,
        geo: VehicleGeometry,
        sampling_time: Timestamp = 1.0,
        params: TrajectoryGeneratorParams = None,
    ):
        self.init_state = init_state
        if params is None:
            self.params = TrajectoryGeneratorParams.default()
        else:
            self.params = params
        start_point = Point((init_state.x, init_state.y))
        # todo only goal position used -> accept position instead of state?
        goal_point = Point((goal_state.x, goal_state.y))
        self.curve_generator = CurveGenerator(start=start_point, goal=goal_point)
        self.geo = geo
        self.sampling_time = sampling_time
        self.dynamics = BicycleDynamics(vg=geo, vp=VehicleParameters.default_car())

    # sample points along A curve -> sample first in curvilinear coo (s) and then (x_n,y_n,v_n,delta_n,theta_n) -> v_n and delta_n are ramdomly sampled
    def sample_points_and_tangents(self, sampling_method: str, curve: BezierCurve):
        n_states = self.params.n_states
        if sampling_method == "random":
            random_ts = [random.random() for _ in range(n_states)]
            random_ts.sort()
        elif sampling_method == "regular":
            random_ts = np.linspace(0, 1, num=n_states)
        else:
            random_ts = []
        random_points = {t: curve.compute_bezier(t=t) for t in random_ts}
        tangents = {t: curve.tangent(t=t) for t in random_ts}
        return random_points, tangents

    # todo find other way rather than random uniform
    def sample_velocity(self):
        return random.uniform(self.params.min_vel, self.params.max_vel)

    def compute_delta(self, theta_1: float, theta_2: float, vel: float, dt: Timestamp):
        delta = (theta_2 - theta_1) * (self.geo.lf + self.geo.lr) / vel / dt
        a = 2
        return np.arctan2(delta, 1.0)

    def states_from_points(
        self, sampling_method: str, curve: BezierCurve
    ) -> List[VehicleState]:  # todo define exact type
        points, tangents = self.sample_points_and_tangents(sampling_method=sampling_method, curve=curve)
        # todo tangent needed?
        # todo initial point should be filled -> here it should be already
        # todo for first it is different
        vehicle_states = [
            VehicleState(x=point[0], y=point[1], theta=curve.theta(t=t), vx=self.sample_velocity(), delta=99999.0)
            for t, point in points.items()
        ]
        vehicle_states[0] = self.init_state  # todo check that this overwriting is correct
        for idx, (state_0, state_1) in enumerate(zip(vehicle_states[:-1], vehicle_states[1:])):
            # todo what velocity to use? velocity incoming?
            delta = self.compute_delta(theta_1=state_0.psi, theta_2=state_1.psi, vel=state_1.vx, dt=self.sampling_time)
            vehicle_states[idx + 1].delta = delta  # todo is this setter?

        return vehicle_states

    # get needed inputs between states (ignoring dynamic effects)
    # todo if not possible, discard
    @staticmethod
    def get_needed_commands(x_0: VehicleState, x_1: VehicleState, dt: Timestamp) -> VehicleCommands:
        acc = (x_1.vx - x_0.vx) / dt
        ddelta = (x_1.delta - x_0.delta) / dt
        return VehicleCommands(acc=acc, ddelta=ddelta)

    def check_feasibility(self, command: VehicleCommands):
        acc_ok = self.params.min_acc <= command.acc <= self.params.max_acc
        ddelta_ok = self.params.min_ddelta <= command.ddelta <= self.params.max_ddelta
        print(acc_ok)
        print(ddelta_ok)
        if not (acc_ok and ddelta_ok):
            print("Not feasible, command: \n")
            print(command)
        return acc_ok and ddelta_ok

    def generate_feasible_trajectory(self, sampling_method: str, curve: BezierCurve):
        states = self.states_from_points(sampling_method=sampling_method, curve=curve)
        commands: List[VehicleCommands] = [VehicleCommands(acc=0, ddelta=0)]  # command to reach initial state
        for state_0, state_1 in zip(states[:-1], states[1:]):
            command = self.get_needed_commands(x_0=state_0, x_1=state_1, dt=self.sampling_time)
            if self.check_feasibility(command):
                commands.append(self.get_needed_commands(x_0=state_0, x_1=state_1, dt=self.sampling_time))
            else:
                return None  # command not feasible -> discard sequence. Resample points

        return states, commands


def transform_points_plotting(points: List[np.ndarray]):
    x = []
    y = []
    for point in points:
        x.append(point[0])
        y.append(point[1])
    return x, y


def test_bezier_curves():
    point_0 = np.array([0, 0])
    point_1 = np.array([5, 0])
    point_2 = np.array([7, 9])
    point_3 = np.array([10, 10])

    points_cubic = [Point(point_0), Point(point_1), Point(point_2), Point(point_3)]
    points_linear = [Point(point_0), Point(point_3)]

    ts = np.linspace(0, 1, num=50)
    bezier_linear = BezierCurve(control_points=points_linear)
    curve_points_linear = [bezier_linear.compute_bezier(t) for t in ts]
    curvature_linear = bezier_linear.max_curvature()
    print(curvature_linear)

    bezier_cubic = BezierCurve(control_points=points_cubic)
    curve_points_cubic = [bezier_cubic.compute_bezier(t) for t in ts]
    curvature_cubic = bezier_cubic.max_curvature()
    print(curvature_cubic)

    x, y = transform_points_plotting(curve_points_linear)
    x_c, y_c = transform_points_plotting(curve_points_cubic)
    fix, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x, y)
    ax.plot(x_c, y_c)
    plt.show()
    return


def test_curve_generator():
    start = Point(np.array([0, 0]))
    goal = Point(np.array([10, 10]))
    curve_gen = CurveGenerator(start=start, goal=goal)

    bezier_curves = curve_gen.get_curve_bundle()

    ts = np.linspace(0, 1, num=50)

    curve_points = []
    for curve in bezier_curves:
        curve_points.append([curve.compute_bezier(t) for t in ts])
    print(len(curve_points))
    plotting_points = [transform_points_plotting(points) for points in curve_points]
    fix, ax = plt.subplots(figsize=(10, 10))
    for points in plotting_points:
        ax.plot(points[0], points[1])
    plt.show()

    return


def test_sampling_points():
    point_0 = np.array([0, 0])
    point_1 = np.array([5, 0])
    point_2 = np.array([7, 9])
    point_3 = np.array([10, 10])

    ts = np.linspace(0, 1, num=50)

    points_cubic = [Point(point_0), Point(point_1), Point(point_2), Point(point_3)]
    points_linear = [Point(point_0), Point(point_3)]
    traj_gen = TrajectoryGenerator(params=TrajectoryGeneratorParams.default())

    bezier_curve = BezierCurve(control_points=points_cubic)

    sampled_points = traj_gen.sample_points_and_tangents(curve=bezier_curve, sampling_method="regular")
    original_points = [bezier_curve.compute_bezier(points_cubic, t) for t in ts]

    x, y = transform_points_plotting(original_points)
    fix, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x, y)

    sampled_points_x = [point[0] for point in sampled_points]
    sampled_points_y = [point[1] for point in sampled_points]

    ax.SCATTER(sampled_points_x, sampled_points_y, marker="o", color="red")
    plt.show()

    print(sampled_points)


def plot_arrows(states: List[VehicleState]):
    dtheta_vectors = []
    ddelta_vectors = []
    for state in states:
        dtheta_vec = np.array((state.vx * np.cos(state.psi), state.vx * np.sin(state.psi)))
        dtheta_norm = np.linalg.norm(dtheta_vec)
        if dtheta_norm == 0.0:
            print(dtheta_vec)
        dtheta_vectors.append(dtheta_vec / dtheta_norm)
        ddelta_vec = np.array((state.vx * np.cos(state.psi + state.delta), state.vx * np.sin(state.psi + state.delta)))
        ddelta_norm = np.linalg.norm(ddelta_vec)
        ddelta_vectors.append(ddelta_vec / ddelta_norm)
    return dtheta_vectors, ddelta_vectors


def test_trajectory_generation():
    sampling_time: Timestamp = 1.0
    initial_state = VehicleState(x=0, y=0, theta=pi / 4, vx=0, delta=pi / 4)
    goal_state = VehicleState(x=10, y=10, theta=0.25, vx=8, delta=0.20)
    vehicle_geometry = VehicleGeometry.default_car()

    traj_gen = TrajectoryGenerator(
        init_state=initial_state, goal_state=goal_state, geo=vehicle_geometry, sampling_time=sampling_time
    )

    start = Point(np.array([0, 0]))
    goal = Point(np.array([10, 10]))
    curve_gen = CurveGenerator(start=start, goal=goal)
    bezier_curves = curve_gen.get_curve_bundle()

    states, commands = traj_gen.generate_feasible_trajectory(sampling_method="regular", curve=bezier_curves[0])
    dtetha_vector, ddelta_vector = plot_arrows(states)

    ts = np.linspace(0, 1, num=50)

    curve_points = []
    curve_points.append([bezier_curves[0].compute_bezier(t) for t in ts])
    plotting_points = [transform_points_plotting(points) for points in curve_points]
    fix, ax = plt.subplots(figsize=(10, 10))
    for points in plotting_points:
        ax.plot(points[0], points[1])

    for idx, vec in enumerate(ddelta_vector):
        plt.arrow(x=states[idx].x, y=states[idx].y, dx=vec[0], dy=vec[1])
    plt.show()

    print(states)
    print(commands)


if __name__ == "__main__":
    test_bezier_curves()
    test_curve_generator()
    test_sampling_points()
    test_trajectory_generation()
