from functools import lru_cache
from math import cos, sin

from shapely import wkt
from shapely.affinity import affine_transform
from shapely.geometry import Polygon

from dg_commons import SE2Transform
from dg_commons.maps import DgLanelet
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from .structures import VehicleTrackState

__all__ = ["get_poly_occupancy", "apply_SE2transform_to_wkt_poly"]


@lru_cache(maxsize=None)
def get_poly_occupancy(vs: VehicleTrackState, vg: VehicleGeometry, ref: DgLanelet) -> Polygon:
    """Gets the rectangles that contain the vehicle."""
    q: SE2Transform = vs.to_global_pose(ref)
    occupancy: Polygon = apply_SE2transform_to_wkt_poly(vg.outline_as_polygon.wkt, q)
    return occupancy


@lru_cache(maxsize=None)
def apply_SE2transform_to_wkt_poly(shapely_geometry_wkt: str, SE2transform: SE2Transform) -> Polygon:
    """Apply SE2 transform to shapely geometry"""
    costheta = cos(SE2transform.theta)
    sintheta = sin(SE2transform.theta)
    coeffs = [costheta, -sintheta, sintheta, costheta, SE2transform.p[0], SE2transform.p[1]]
    shapely_geometry = wkt.loads(shapely_geometry_wkt)
    return affine_transform(shapely_geometry, coeffs)


### The following is not used anymore, but kept for reference
# @dataclass(frozen=True)
# class PolygonHashable:
#     points: Tuple[Tuple[float, float]]
#
#     def as_polygon(self) -> Polygon:
#         return Polygon(self.points)
#
#     @classmethod
#     def from_polygon(cls, polygon: Polygon) -> "PolygonHashable":
#         assert polygon.is_valid
#         return cls(points=tuple(p for p in polygon.exterior.coords))
#
#
# def poly_resources_checker(a: FSet[PolygonHashable], b: FSet[PolygonHashable]) -> bool:
#     """Do two future resources intersect?"""
#     return any(pa.as_polygon().intersects(pb.as_polygon()) for pa in a for pb in b)
