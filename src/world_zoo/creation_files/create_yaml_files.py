import numpy as np
import subprocess
import yaml
from typing import Dict, Union, List

Coordinate = Union[float, int]
LaneWidth = Union[float, int]
LaneID = Union[int, str]
Order = int
SplineSpec = Dict[str, Union[List[Coordinate], LaneWidth, Order]]

def copy2clip(txt: str) -> int:
    """Copies string to clipboard (xclip has to be installed)"""
    cmd='echo '+txt.strip()+' | xclip -i'
    return subprocess.check_call(cmd, shell=True)

def rot_m(theta: float, radians: bool = False) -> np.array:
    """Returns rotation matrix (for a left hand coordinate system)"""
    if not radians:
        theta = np.radians(theta)
    c, s = np.cos(-theta), np.sin(-theta)
    return np.array(((c, -s), (s, c)))

def ctr_p_dict_intersection() -> SplineSpec:
    """Creates the spline specifications for the intersection (YAML format)"""
    x_straight1 = [38.4, 38.4]
    y_straight1 = [0, 84]

    x_straight2 = [38.4, 38.4]
    y_straight2 = [0, 84]

    x_straight3 = [0, 84]
    y_straight3 = [45.6, 45.6]

    x_straight4 = [45.6, 45.6]
    y_straight4 = [0, 84]



    half_length = 150  # control points number for left and right turn will be double of this value

    x1_right = [round(float(_),2) for _ in np.linspace(84, 42.6, half_length)]
    x2_right = [45.6]*half_length
    x_right1 = x1_right + x2_right

    y1_right = [38.4] * half_length
    y2_right = [round(float(_), 2) for _ in np.linspace(35.4, 0, half_length)]
    y_right1 = y1_right + y2_right

    cord_right1_np = np.array([x_right1, y_right1])

    x1_left = [round(float(_),2) for _ in np.linspace(84, 45.6, half_length)]
    x2_left = [38.4]*half_length
    x_left1 = x1_left + x2_left

    y1_left = [38.4] * half_length
    y2_left = [round(float(_), 2) for _ in np.linspace(38.4, 84, half_length)]
    y_left1 = y1_left + y2_left

    cord_left1_np = np.array([x_left1, y_left1])

    trans = np.array([[-42],[-42]])

    theta = 90
    cord_right2_np = np.matmul(rot_m(theta), (cord_right1_np + trans)) - trans

    x_right2 = [round(float(_),2) for _ in cord_right2_np[0]]
    y_right2 = [round(float(_),2) for _ in cord_right2_np[1]]

    cord_left2_np = np.matmul(rot_m(theta), (cord_left1_np + trans)) - trans

    x_left2 = [round(float(_), 2) for _ in cord_left2_np[0]]
    y_left2 = [round(float(_), 2) for _ in cord_left2_np[1]]

    theta = 180
    cord_right3_np = np.matmul(rot_m(theta), (cord_right1_np + trans)) - trans

    x_right3 = [round(float(_), 2) for _ in cord_right3_np[0]]
    y_right3 = [round(float(_), 2) for _ in cord_right3_np[1]]

    cord_left3_np = np.matmul(rot_m(theta), (cord_left1_np + trans)) - trans

    x_left3 = [round(float(_), 2) for _ in cord_left3_np[0]]
    y_left3 = [round(float(_), 2) for _ in cord_left3_np[1]]

    theta = 270
    cord_right4_np = np.matmul(rot_m(theta), (cord_right1_np + trans)) - trans

    x_right4 = [round(float(_), 2) for _ in cord_right4_np[0]]
    y_right4 = [round(float(_), 2) for _ in cord_right4_np[1]]

    cord_left4_np = np.matmul(rot_m(theta), (cord_left1_np + trans)) - trans

    x_left4 = [round(float(_), 2) for _ in cord_left4_np[0]]
    y_left4 = [round(float(_), 2) for _ in cord_left4_np[1]]

    yml_dict = {
        '1-straight': {
            'x': x_straight1,
            'y': y_straight1,
            'w': 6,
            'order': 1
        },
        '1-right': {
                'x': x_right1,
                'y': y_right1,
                'w': 6,
                'order': 2
            },
        '1-left': {
            'x': x_left1,
            'y': y_left1,
            'w': 6,
            'order': 3
        },
        '2-straight': {
            'x': x_straight2,
            'y': y_straight2,
            'w': 6,
            'order': 1
        },
        '2-right': {
            'x': x_right2,
            'y': y_right2,
            'w': 6,
            'order': 2
        },
        '2-left': {
            'x': x_left2,
            'y': y_left2,
            'w': 6,
            'order': 3
        },
        '3-straight': {
            'x': x_straight3,
            'y': y_straight3,
            'w': 6,
            'order': 1
        },
        '3-right': {
            'x': x_right3,
            'y': y_right3,
            'w': 6,
            'order': 2
        },
        '3-left': {
            'x': x_left3,
            'y': y_left3,
            'w': 6,
            'order': 3
        },
        '4-straight': {
            'x': x_straight4,
            'y': y_straight4,
            'w': 6,
            'order': 1
        },
        '4-right': {
            'x': x_right4,
            'y': y_right4,
            'w': 6,
            'order': 2
        },
        '4-left': {
            'x': x_left4,
            'y': y_left4,
            'w': 6,
            'order': 3
        }
    }
    return yml_dict

def ctr_p_dict_10_lane_highway() -> SplineSpec:
    pass

if __name__ == '__main__':
    # create intersection yml file
    fname = "intersection.yml"
    with open(fname, 'w') as yml_file:
        yaml.dump(ctr_p_dict_intersection(), yml_file, default_flow_style=None)

