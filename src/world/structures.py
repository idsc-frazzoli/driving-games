from dataclasses import dataclass
from typing import FrozenSet, Union, Dict, List, Sequence, Any, Tuple
from numbers import Number
from matplotlib import pyplot as plt
from decimal import Decimal as D
import yaml
import numpy as np
from scipy import interpolate
from contextlib import contextmanager


__all__ = ["load_world", "Lane", "World"]


@dataclass(frozen=True)
class Lane:
    ctr_p_x: Sequence[Number]
    ctr_p_y: Sequence[Number]
    w: Number
    order: int

    @property
    def control_points(self) -> List[Sequence[Number]]:
        return [self.ctr_p_x, self.ctr_p_y]

    @property
    def control_points_as_D(self) -> List[Sequence[D]]:
        return [tuple(map(D, self.ctr_p_x)), tuple(map(D, self.ctr_p_y))]

    @property
    def tck(self):
        ctr_p = self.control_points
        if ctr_p[0][0] == ctr_p[0][-1] and ctr_p[1][0] == ctr_p[1][-1]:
            tck, _ = interpolate.splprep(ctr_p, k=self.order, per=True)
        else:
            tck, _ = interpolate.splprep(ctr_p, k=self.order)
        return tck

    @property
    def start_position(self) -> List[np.array]:
        return self.get_position(0)

    @property
    def end_position(self) -> List[np.array]:
        return self.get_position(1)

    def get_position(self, s: Union[Sequence[Number], Number]) -> List[np.array]:
        return interpolate.splev(s, self.tck)

    def get_derivative(self, s: Union[Sequence[Number], Number], deriv_order: int) -> List[np.array]:
        return interpolate.splev(s, self.tck, der=deriv_order)

class Image:
    pass


@dataclass(frozen=True)
class World:
    name: str
    lanes: Tuple[Lane]
    background: Union[np.ndarray, Image]
    scale: float

    def plot_world(self):
        # todo[Chris] Add Cars and boundaries
        fig, ax = plt.subplots()
        # fig.tight_layout()
        ax.set_title(self.name)
        ax.imshow(self.background)
        s = np.linspace(0, 1, 1000)
        for lane in self.lanes:
            middle_line = [_ * self.scale for _ in lane.get_position(s)]
            ax.plot(middle_line[0], middle_line[1])
            dx, dy = lane.get_derivative(s, deriv_order=1)
            #normal = np.column_stack([-dy, dx])
            #normal_unit = normal / np.linalg.norm(normal, axis=1)[:, np.newaxis]
            #left_bound = middle_line - lane.w * normal_unit
            #right_bound = middle_line + lane.w * normal_unit
            #ax.plot(left_bound[:, 0], left_bound[:, 1], '-.r', linewidth=1.0, label='Left bound')
            #ax.plot(right_bound[:, 0], right_bound[:, 1], '--r', linewidth=1.0, label='Right bound')

        fig.show()
        plt.close(fig=fig)


def load_world(
        name: str,  # name of world
        background_path: str,  # path of background picture
        scale: float,  # [pixels/meter]
        control_points_path: str  # path of spline definition file
) -> World:
    background = plt.imread(background_path)
    parsed_yaml_file: Dict[List[Sequence[Number], Number]]
    # open and parse the YAML file
    with open(control_points_path) as yml_file:
        parsed_yaml_file = yaml.load(yml_file, Loader=yaml.FullLoader)
    lanes = []
    # Check if all lanes are fully specified in YAML file
    assert len(set(map(len, parsed_yaml_file.values()))) == 1, \
        "Missing specification for one or more lanes in YAML file!"
    for (x, y, w, order) in zip(*parsed_yaml_file.values()):
        x = tuple(x)  # make control points immutable
        y = tuple(y)
        lanes.append(Lane(ctr_p_x=x, ctr_p_y=y, w=w, order=order))

    return World(
        name=name,
        lanes=tuple(lanes),
        background=background,
        scale=scale
    )
