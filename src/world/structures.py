from dataclasses import dataclass
from typing import Union, Dict, List, Sequence, Tuple, Optional
from numbers import Number
from matplotlib import pyplot as plt
from decimal import Decimal as D
import yaml
import numpy as np
from scipy import interpolate
import os


__all__ = ["load_world", "Lane", "World"]

Coordinate = Union[float, int]
LaneWidth = Union[float, int]
LaneID = Union[int, str]
SplineSpec = Dict[str, Union[List[Coordinate], LaneWidth]]


@dataclass(frozen=True)
class Lane:
    lane_id: LaneID  # identifier of lane
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
            tck, *u = interpolate.splprep(ctr_p, k=self.spl_order, per=True)  # fixme why warning without *

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
    # todo define class image
    pass


@dataclass(frozen=True)
class World:
    name: str  # Name of the world
    lanes: Tuple[Lane]  # Collection of lane objects
    background: Union[np.ndarray, Image]  # Bitmap of background
    scale: float  # [pixel/meter]

    def plot_world(self, save_png_path: Optional[str] = None):
        fig, ax = plt.subplots()
        ax.set_title(self.name)
        ax.imshow(self.background)
        s = np.linspace(0, 1, 1000)
        for lane in self.lanes:
            middle_line = [_ * self.scale for _ in lane.get_position(s)]

            dx, dy = lane.get_derivative(s, deriv_order=1)
            normal = np.column_stack([-dy, dx])
            normal_unit = normal / np.linalg.norm(normal, axis=1)[:, np.newaxis]
            left_bound = np.column_stack(middle_line) - lane.w * self.scale / 2 * normal_unit
            right_bound = np.column_stack(middle_line) + lane.w * self.scale / 2 * normal_unit

            ax.plot(middle_line[0], middle_line[1], '-b', linewidth=0.5, label='Middle Line')
            ax.plot(left_bound[:, 0], left_bound[:, 1], '--r', linewidth=1.0, label='Left bound')
            ax.plot(right_bound[:, 0], right_bound[:, 1], '--r', linewidth=1.0, label='Right bound')

        if save_png_path:
            ax.set_axis_off()
            mkdir_p(save_png_path)
            fig.savefig(save_png_path + self.name, dpi=fig.dpi, bbox_inches='tight')

        fig.tight_layout()
        fig.show()
        plt.close(fig=fig)


def load_world(
        name: str,  # name of world
        background_path: str,  # path of background picture
        control_points_path: str,  # path of spline definition file
        scale: Optional[float] = None,  # [pixels/meter]
        ref_lane_px: Optional[int] = None  # pixels of reference lane
) -> World:
    if not bool(scale) ^ bool(ref_lane_px):
        msg = "Either scale or ref_lane_px has to be different from None. But not both!"
        raise ValueError(msg)

    background = plt.imread(background_path)
    lanes = get_lanes(control_points_path=control_points_path)
    if not scale:
        scale = ref_lane_px / lanes[0].w
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

    lanes: List[Lane] = []
    _id: LaneID
    ctr_x: List[Coordinate]
    ctr_y: List[Coordinate]
    w: LaneWidth
    spl_order: int
    for _id in parsed_yaml_file:
        ctr_x = parsed_yaml_file[_id]["x"]
        ctr_y = parsed_yaml_file[_id]["y"]
        w = parsed_yaml_file[_id]["w"]
        spl_order = parsed_yaml_file[_id]["order"]
        # Check if all lanes are fully specified in YAML file
        check_spl_params(_id=_id, ctr_x=ctr_x, ctr_y=ctr_y, w=w, spl_order=spl_order)
        lanes.append(Lane(lane_id=_id, ctr_p_x=ctr_x, ctr_p_y=ctr_y, w=w, spl_order=spl_order))
    return lanes


def check_spl_params(
        _id: LaneID, ctr_x: List[Coordinate],
        ctr_y: List[Coordinate],
        w: LaneWidth,
        spl_order: int
) -> None:
    msg = f"Lane {_id} is not fully specified! Control point sequences do not have the same length"
    assert len(ctr_x) == len(ctr_y), msg
    msg = f"Lane {_id} is not fully specified! Lane width has to be a single integer or float"
    assert type(w) == int or type(w) == float, msg
    msg = f"Lane {_id} is not fully specified! Spline order has to be a single integer"
    assert type(spl_order) == int, msg

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise

