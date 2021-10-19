from dataclasses import dataclass
from dg_commons import PlayerName, SE2Transform
from dg_commons.sim import PlayerObservations
from typing import MutableMapping, Dict, Optional
from shapely.geometry import Polygon
import numpy as np
from dg_commons.geo import SE2_apply_T2, T2value


@dataclass
class SituationObservations:
    my_name: PlayerName

    agents: Optional[MutableMapping[PlayerName, PlayerObservations]] = None

    rel_poses: Optional[Dict[PlayerName, SE2Transform]] = None

    dt_commands: Optional[float] = None


def front_polygon(my_occupancy: Polygon, dist: float):
    boundaries = my_occupancy.exterior.coords
    pos1, pos2 = boundaries[0], boundaries[1]
    pos4_p, pos3_p = np.array(boundaries[2]) - np.array(pos2), np.array(boundaries[3]) - np.array(pos1)
    pos3 = tuple(np.array(boundaries[2]) + pos4_p / np.linalg.norm(pos4_p) * dist)
    pos4 = tuple(np.array(boundaries[3]) + pos4_p / np.linalg.norm(pos4_p) * dist)
    polygon = Polygon((pos1, pos2, pos3, pos4, pos1))
    return polygon


def relative_velocity(my_vel: float, other_vel: float, transform):
    other_vel_wrt_other = [float(other_vel), 0.0]
    other_vel_wrt_myself = SE2_apply_T2(transform, other_vel_wrt_other)
    return my_vel - other_vel_wrt_myself[0]
