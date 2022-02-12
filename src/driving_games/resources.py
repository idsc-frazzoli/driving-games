from dataclasses import dataclass
from typing import FrozenSet as FSet, Tuple

from geometry import SE2value
from shapely.geometry import Polygon

from dg_commons import apply_SE2_to_shapely_geo
from dg_commons.maps import DgLanelet
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from driving_games import VehicleTrackState


@dataclass(frozen=True)
class PolygonHashable:
    points: Tuple[Tuple[float, float]]

    def as_polygon(self) -> Polygon:
        return Polygon(self.points)

    @classmethod
    def from_polygon(cls, polygon: Polygon) -> "PolygonHashable":
        assert polygon.is_valid
        return cls(points=tuple(p for p in polygon.exterior.coords))


def get_resources_used(vs: VehicleTrackState, vg: VehicleGeometry, ref: DgLanelet, ds: float) -> FSet[PolygonHashable]:
    """Gets the rectangles that contain the vehicle."""
    q: SE2value = vs.to_global_pose(ref).as_SE2()
    occupancy: Polygon = apply_SE2_to_shapely_geo(vg.outline_as_polygon, q).buffer(ds)
    # todo this for now is an approximation,
    #  we need to compute the forward resources given the state and commands (or the future positions)
    return frozenset([PolygonHashable.from_polygon(occupancy)])


def poly_resources_checker(a: FSet[PolygonHashable], b: FSet[PolygonHashable]) -> bool:
    return any(pa.as_polygon().intersects(pb.as_polygon()) for pa in a for pb in b)
