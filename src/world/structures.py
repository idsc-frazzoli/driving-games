from dataclasses import dataclass
from typing import NewType, Union, Dict, List, Sequence, Tuple, FrozenSet
from numbers import Number
from matplotlib import pyplot as plt
from decimal import Decimal as D
import yaml
import numpy as np
from scipy import interpolate


__all__ = ["load_world", "Lane", "World"]

# Coordinate = NewType('Coordinate', float)
# LaneWidth = NewType('LaneWidth', float)
# LaneID = NewType('LaneID', Union[int,str])
# SplineSpec = NewType('SplineSpec', Dict[str, Union[List[Coordinate], LaneWidth]])
Coordinate = Union[float, int]
LaneWidth = Union[float, int]
LaneID = Union[int, str]
SplineSpec = Dict[str, Union[List[Coordinate], LaneWidth]]

@dataclass(frozen=True)
class Lane:
    id: LaneID  # identifier of lane
    ctr_p_x: Sequence[Coordinate]  # x-coordinate of control-points
    ctr_p_y: Sequence[Coordinate]  # y-coordinate of control-points
    w: LaneWidth  # [m]
    spl_order: int  # order of spline

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
            tck, *u = interpolate.splprep(ctr_p, k=self.spl_order, per=True)  #fixme why warning without *

        else:
            tck, *u = interpolate.splprep(ctr_p, k=self.spl_order)
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
    scale: float # [pixel/meter]

    def plot_world(self):
        # todo[Chris] Add Cars
        fig, ax = plt.subplots()
        fig.tight_layout()
        ax.set_title(self.name)
        ax.imshow(self.background)
        s = np.linspace(0, 1, 1000)
        for lane in self.lanes:
            middle_line = [_ * self.scale for _ in lane.get_position(s)]

            dx, dy = lane.get_derivative(s, deriv_order=1)
            normal = np.column_stack([-dy, dx])
            normal_unit = normal / np.linalg.norm(normal, axis=1)[:, np.newaxis]
            left_bound = np.column_stack(middle_line) - lane.w * self.scale * normal_unit
            right_bound = np.column_stack(middle_line) + lane.w * self.scale * normal_unit

            ax.plot(middle_line[0], middle_line[1],':b', linewidth=1.0, label='Middle Line')
            ax.plot(left_bound[:, 0], left_bound[:, 1], '-.r', linewidth=1.0, label='Left bound')
            ax.plot(right_bound[:, 0], right_bound[:, 1], '--r', linewidth=1.0, label='Right bound')

        fig.show()
        plt.close(fig=fig)


def load_world(
        name: str,  # name of world
        background_path: str,  # path of background picture
        scale: float,  # [pixels/meter]
        control_points_path: str  # path of spline definition file
) -> World:

    background = plt.imread(background_path)

    lanes = get_lanes(control_points_path=control_points_path)

    return World(
        name=name,
        lanes=tuple(lanes),
        background=background,
        scale=scale
    )

def get_lanes(control_points_path: str) -> List[Lane]:
    parsed_yaml_file: Dict[LaneID, SplineSpec]
    # open and parse the YAML file
    with open(control_points_path) as yml_file:
        parsed_yaml_file = yaml.load(yml_file, Loader=yaml.FullLoader)
    lanes = []


    ctr_x: List[Coordinate]
    ctr_y: List[Coordinate]
    w: LaneWidth
    for id in parsed_yaml_file:
        ctr_x = parsed_yaml_file[id]['x']
        ctr_y = parsed_yaml_file[id]['y']
        w = parsed_yaml_file[id]['w']
        order = parsed_yaml_file[id]['order']
        # Check if all lanes are fully specified in YAML file
        msg = f"Lane {id} is not fully specified!"
        assert len(ctr_x) == len(ctr_y) and len(list([w])) == 1 and type(order) == int, msg
        lanes.append(Lane(id=id, ctr_p_x=ctr_x, ctr_p_y=ctr_y, w=w, spl_order=order))
    return lanes
