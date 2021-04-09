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
    """ The grid cell size """

    total_nb_of_cells: int
    """ The total number of grid cells on the map """

    nb_H: int
    """ The height of the map in number of grid cells """

    nb_W: int
    """ The width of the map in number of grid cells"""

    resources: Mapping[ResourceID, CenterPointGridCell]
    """The shared resources. To each coordinate of the center point of the cell a unique ID has been assigned"""

    def __init__(self, cell_size: D, *args, **kwargs):
        self.cell_size = cell_size
        DuckietownMap.__init__(self, *args, **kwargs)  # initialize the duckietwon map

        H = self['tilemap'].H  # get the height in numbers of tiles
        W = self['tilemap'].W  # get the width in numbers of tiles

        # get the number of resource cells for each direction
        nb_H = math.ceil(self.tile_size * H / cell_size)
        nb_W = math.ceil(self.tile_size * W / cell_size)
        self.nb_H = nb_H
        self.nb_W = nb_W
        self.nb_of_cells = nb_H * nb_W  # total number of cells on the map

        x0 = y0 = cell_size / D(2)  # get the coordinate of the center point first cell on the map

        r0 = Coordinates((x0, y0))  # transform to coordinates
        vec_x = Coordinates((cell_size, D(0)))  # get the vector pointing to the next cell in x direction
        vec_y = Coordinates((D(0), cell_size))  # get the vector pointing to the next cell in y direction
        resources: MutableMapping[ResourceID, CenterPointGridCell] = {}
        _id = 0
        for j in range(nb_H):
            for i in range(nb_W):
                # get the coordinates from the bottom left to the top right of the center point of the grid cells
                resources[_id] = r0 + i * vec_x + j * vec_y
                _id += 1

        self.resources = frozendict(resources)

    @classmethod
    def initializor(cls, m: DuckietownMap, cell_size: D) -> "DrivingGameGridMap":
        """
        Transform a DuckieTown map to a DrivingGameGridMap
        """
        ls_dict = m.__dict__
        return cls(cell_size=cell_size, **ls_dict)

    @memoized_reset
    def get_resources_used(self, vs: DuckieState, vg: DuckieGeometry) -> FrozenSet[ResourceID]:
        """
        For a certain state of a player get the resources used by upsampling in front and in the back of the car.
        """

        dt = D(0.5)  # todo other timestep than 1
        n = 2
        xs = sample_x(vs.x, vs.v, dt=dt, n=n)  # upsample in front and the back of the car along the lane

        resources_id_used: Set[ResourceID] = set()
        for x in xs:  # get the resource of the footprint for the different positions along the lane
            resources_id_used |= self.get_resource_footprint_from_along_lane(lane=vs.lane, along_lane=x, vg=vg)

        return frozenset(resources_id_used)

    @memoized_reset
    def get_resource_footprint_from_along_lane(
            self,
            lane: Lane,
            along_lane: D,
            vg: DuckieGeometry
    ) -> FrozenSet[ResourceID]:
        """
        This function returns the resource IDs of the grid cells that lie inside the cars footprint
        """
        cell_size = self.cell_size
        nb_W = self.nb_W

        # get the range around the car that has to be checked (corresponds to the length of the diagonal of the car
        # in numbers of grid cells)
        range_to_check = self._cell_span_to_check(vg=vg)
        resources_id_used: Set[ResourceID] = set()

        # get the rectangle of the car
        rect_car = projected_car_from_along_lane(lane=lane, along_lane=along_lane, vg=vg).rectangle
        car_center = rect_car.center  # get the center pose of the car
        # get the grid cell ID of the center point of the car
        car_center_point_id = int(car_center[0] // cell_size) + int(car_center[1] // cell_size) * nb_W
        if range_to_check == 1:  # happens if the grid size is bigger than the rectangle of the car
            resources_id_used.add(car_center_point_id)
        else:
            # get the grid cell ID where we have to start checking if the cells lie inside the cars rectangle
            # start_ID = car_center_point_id - (range_to_check // 2) - (range_to_check // 2) * nb_W
            start_ID = car_center_point_id - (nb_W + 1) * (range_to_check // 2)  # simplified version of above

            # Check the "grid cell rectangle" around the car centered at the midpoint of the cars rectangle
            # with length (in number of cells) of the diagonal of the cars rectangle, if any of the grid cells mid point
            # lie inside of the cars rectangle.
            for j in range(range_to_check):
                for i in range(range_to_check):
                    _id = start_ID + i + j * nb_W
                    try:
                        center_point_gc = self.resources[_id]  # get the coordinate of the center point of the grid cell
                        if rect_car.contains_point(center_point_gc):  # check if it lies inside the cars rectangle
                            resources_id_used.add(_id)
                    except KeyError:
                        # Resource not on map anymore
                        pass
        return frozenset(resources_id_used)

    def _cell_span_to_check(self, vg: DuckieGeometry) -> int:
        """
        Computes the length of the diagonal of the cars rectangle in number of grid cells (always an odd number)
        """
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
