from decimal import Decimal
from typing import FrozenSet as FSet, Tuple, Dict

import numpy as np
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from cytoolz import sliding_window
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from dg_commons import SE2_apply_T2
from dg_commons.maps import DgLanelet

__all__ = ["ResourcesOccupancy", "CellID"]

CellID = Tuple[int, int]


class ResourcesOccupancy:
    def __init__(self, lanelet_network: LaneletNetwork, cell_resolution: Decimal):
        self.lanelet_network = lanelet_network
        """the commonroad lanelet network"""
        self.cell_resolution = float(cell_resolution)
        """corresponds to the max/average subdivision of the cells in the rtree"""
        self.strtree: STRtree = self._create_rtree()

    def get_occupied_cells(
        self,
    ) -> FSet[CellID]:
        pass

    def _create_rtree(self) -> STRtree:
        """Creates a subdivision of the lanes in smaller pieces storing them in a rtree structure"""
        resources: Dict[CellID, Polygon] = {}

        for lanelet in self.lanelet_network.lanelets:
            lanelet_res = self._subdivide_lanelet_into_polygons(lanelet)
            resources.update(lanelet_res)

        res_idx, res_poly = zip(*resources.items())
        srtree = STRtree(geoms=res_poly, items=res_idx, node_capacity=3)
        return srtree

    def _subdivide_lanelet_into_polygons(self, lanelet: Lanelet) -> Dict[CellID, Polygon]:
        dg_lanelet = DgLanelet.from_commonroad_lanelet(lanelet)
        lanelet_length = dg_lanelet.get_lane_length()
        n_polygons: int = int(lanelet_length // self.cell_resolution)
        resources: Dict[CellID, Polygon] = {}

        def get_left_right_point(along_lane_: float) -> Tuple[np.ndarray, np.ndarray]:
            beta = dg_lanelet.beta_from_along_lane(along_lane_)
            q = dg_lanelet.center_point(beta)
            r = dg_lanelet.radius(beta)
            delta_left = np.array([0, r])
            delta_right = np.array([0, -r])
            left = SE2_apply_T2(q, delta_left)
            right = SE2_apply_T2(q, delta_right)
            return left, right

        along_lane_values = np.linspace(0, lanelet_length, num=n_polygons, endpoint=True)
        for j, along_lane in enumerate(sliding_window(2, along_lane_values)):
            along_lane_0, along_lane_1 = along_lane
            left0, right0 = get_left_right_point(along_lane_0)
            left1, right1 = get_left_right_point(along_lane_1)
            poly = Polygon([left0, left1, right1, right0, left0])
            id_poly = {(lanelet.lanelet_id, j): poly}
            resources.update(id_poly)

        return resources


def cells_resources_checker(a: FSet[CellID], b: FSet[CellID]) -> bool:
    """Do two future resources intersect?"""
    return bool(a & b)
