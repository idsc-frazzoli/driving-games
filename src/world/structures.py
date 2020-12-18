from dataclasses import dataclass
from typing import FrozenSet, Union, Dict, List
from matplotlib import pyplot as plt
import numpy as np
import os
import yaml


__all__ = ["load_world"]

path=os.path.dirname(__file__)

@dataclass
class Lane:
    x: List[int]
    y: List[int]
    w: int

class Image:
    pass

@dataclass(frozen=True)
class World:
    name: str
    lanes: FrozenSet[Lane]
    background: Union[np.ndarray, Image]
    scale: float

def load_world(
        name: str,
        background_path: str,
        scale: float,
        control_points_path: str
) -> World:
    background = plt.imread(background_path)
    with open(control_points_path) as yml_file:
        parsed_yaml_file = yaml.load(yml_file, Loader=yaml.FullLoader)
    lanes = []
    for (x, y, w) in zip(*parsed_yaml_file.values()):
        lanes.append(Lane(x=x,y=y,w=w))

    return World(
        name=name,
        lanes=lanes,
        background=background,
        scale=scale
    )

