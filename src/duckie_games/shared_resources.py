import math

from frozendict import frozendict
from decimal import Decimal as D, localcontext
from typing import Mapping, MutableMapping, Set, FrozenSet

from duckietown_world.world_duckietown.duckietown_map import DuckietownMap
from duckietown_world.utils import memoized_reset

from world.utils import Lane

from duckie_games.structures import DuckieState, DuckieGeometry
from duckie_games.rectangle import Coordinates, projected_car_from_along_lane, sample_x

ResourceID = int
CenterPointGridCell = Coordinates


class DrivingGameGridMap(DuckietownMap):
    """
    Wrapper class for a Duckietown Map containing a discretized resource-grid.
    """
    cell_size: D
    total_nb_of_cells: int
    nb_H: int
    nb_W: int
    resources: Mapping[ResourceID, CenterPointGridCell]

    def __init__(self, cell_size: D, *args, **kwargs):
        self.cell_size = cell_size
        DuckietownMap.__init__(self, *args, **kwargs)

        H = self['tilemap'].H
        W = self['tilemap'].W

        nb_H = math.ceil(self.tile_size * H / cell_size)
        nb_W = math.ceil(self.tile_size * W / cell_size)
        self.nb_H = nb_H
        self.nb_W = nb_W
        self.nb_of_cells = nb_H * nb_W

        x0 = y0 = cell_size / D(2)

        r0 = Coordinates((x0, y0))
        vec_x = Coordinates((cell_size, D(0)))
        vec_y = Coordinates((D(0), cell_size))
        resources: MutableMapping[ResourceID, CenterPointGridCell] = {}
        _id = 0
        for j in range(nb_H):
            for i in range(nb_W):
                resources[_id] = r0 + i * vec_x + j * vec_y
                _id += 1

        self.resources = frozendict(resources)

    @classmethod
    def initializor(cls, m: DuckietownMap, cell_size: D) -> "DrivingGameGridMap":
        ls_dict = m.__dict__
        return cls(cell_size=cell_size, **ls_dict)

    @memoized_reset
    def get_resources_used(self, vs: DuckieState, vg: DuckieGeometry) -> FrozenSet[ResourceID]:

        dt = D(0.5)
        n = 2
        xs = sample_x(vs.x, vs.v, dt=dt, n=n)

        resources_id_used: Set[ResourceID] = set()
        for x in xs:
            resources_id_used |= self.get_resource_footprint_from_along_lane(lane=vs.lane, along_lane=x, vg=vg)

        return frozenset(resources_id_used)

    @memoized_reset
    def get_resource_footprint_from_along_lane(
            self,
            lane: Lane,
            along_lane: D,
            vg: DuckieGeometry
    ) -> FrozenSet[ResourceID]:
        cell_size = self.cell_size
        nb_W = self.nb_W

        range_to_check = self._cell_span_to_check(vg=vg)
        resources_id_used: Set[ResourceID] = set()

        rect_car = projected_car_from_along_lane(lane=lane, along_lane=along_lane, vg=vg).rectangle
        car_center = rect_car.center
        car_center_point_id = int(car_center[0] // cell_size) + int(car_center[1] // cell_size) * nb_W
        if range_to_check == 1:
            resources_id_used.add(car_center_point_id)
        else:
            # start_ID = car_center_point_id - (range_to_check // 2) - (range_to_check // 2) * nb_W
            start_ID = car_center_point_id - (nb_W + 1) * (range_to_check // 2)  # simplified version of above
            for j in range(range_to_check):
                for i in range(range_to_check):
                    _id = start_ID + i + j * nb_W
                    try:
                        center_point_gc = self.resources[_id]
                        if rect_car.contains_point(center_point_gc):
                            resources_id_used.add(_id)
                    except KeyError:
                        # Resource not on map anymore
                        pass
        return frozenset(resources_id_used)

    def _cell_span_to_check(self, vg: DuckieGeometry) -> int:
        with localcontext() as ctx:
            ctx.prec = 2
            car_diag_squared = pow(vg.length, 2) + pow(vg.width, 2)
            car_diag = car_diag_squared.sqrt()
            length_in_gc = math.ceil(car_diag / self.cell_size)
            if length_in_gc % 2 == 0:
                res = length_in_gc + 1
            else:
                res = length_in_gc
        return res
