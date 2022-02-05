from typing import FrozenSet

from shapely.geometry import Polygon

from dg_commons import SE2Transform, apply_SE2_to_shapely_geo
from dg_commons.maps import DgLanelet
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from driving_games import VehicleTrackState


def get_resources_used(vs: VehicleTrackState, vg: VehicleGeometry, ref: DgLanelet, ds: float) -> FrozenSet[Polygon]:
    """Gets the rectangles that contain the vehicle."""
    q: SE2Transform = vs.to_global_pose(ref)
    occupancy: Polygon = apply_SE2_to_shapely_geo(vg.outline_as_polygon, q).inflate(ds)
    # todo this for now is an approximation,
    #  we need to compute the forward resources given the state and commands (or the future positions)
    return frozenset([occupancy])
