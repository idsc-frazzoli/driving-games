from dataclasses import dataclass
from decimal import Decimal
from typing import NewType, Mapping

from commonroad_dc import pycrcc
from geometry import SE2_from_xytheta, SE2value

import numpy as np

from sim.models.vehicle import VehicleState, VehicleGeometry, VehicleModel

ImpactLocation = NewType("ImpactLocation", str)
IMPACT_FRONT = ImpactLocation("front")
IMPACT_BACK = ImpactLocation("back")
IMPACT_LEFT = ImpactLocation("left")
IMPACT_RIGHT = ImpactLocation('right')


class CollisionReport:

    def __init__(self):
        self.collision: bool = False
        self.location: Mapping[ImpactLocation, pycrcc.Triangle]
        self.active: bool
        self.rel_velocity: float
        self.energy_transfer: float

    def set_impact_location(self, impact_location: Mapping[ImpactLocation, pycrcc.Triangle]):
        self.location = impact_location
        self.collision = True

    def set_rel_velocity(self, a_speed: float, b_speed: float):
        # todo : not just subtract but project in proper RF
        self.rel_velocity = abs(a_speed - b_speed)

    def set_energy_transfer(self):
        # todo: properly implement this
        self.energy_transfer = 0.0



def get_mesh(vehicle: VehicleModel) -> Mapping[ImpactLocation, pycrcc.Triangle]:
    """
    This returns all the vertices of a rectangle in the global reference frame of the map (a bit useless for now)
    :param vehicle:
    :return:
    """

    vertices = vehicle.get_vertices()
    state    = vehicle.get_state()

    impactlocations: Mapping[ImpactLocation, pycrcc.Triangle] = {
        IMPACT_FRONT: pycrcc.Triangle(*vertices[:, 0], *vertices[:, 1], (state.x, state.y)),
        IMPACT_BACK:  pycrcc.Triangle(*vertices[:, 2], *vertices[:, 3], (state.x, state.y)),
        IMPACT_LEFT:  pycrcc.Triangle(*vertices[:, 1], *vertices[:, 2], (state.x, state.y)),
        IMPACT_RIGHT: pycrcc.Triangle(*vertices[:, 3], *vertices[:, 0], (state.x, state.y)),
    }

    return impactlocations


def compute_collision_report(a: VehicleModel, b: VehicleModel) -> CollisionReport:
    report = CollisionReport()

    # a_mesh = get_mesh(a)
    b_mesh = get_mesh(b)

    for key, zone in b_mesh.items():
        tmp = a.get_footprint().collide(zone)

        if tmp:
            report.set_impact_location({key: zone})
            report.set_rel_velocity(a.get_state().vx, b.get_state().vx)
            report.set_energy_transfer()
            print(f"Vehicle X collided with {key} from car Y")

    return report


"""
def get_vertices_g(rect: pycrcc.RectOBB) -> np.ndarray:

This gets the car vertices in the global reference frame (RF)
:param rect:
:return:


l2g: SE2value = SE2_from_xytheta((rect.center()[0], rect.center()[1], rect.orientation()))

vertices = np.array([[rect.r_x(), -rect.r_x(), -rect.r_x(), rect.r_x()],
                     [rect.r_y(), rect.r_y(), -rect.r_y(), -rect.r_y()],
                     [1, 1, 1, 1]])

vertices = l2g @ vertices

vertices = vertices[:-1, :]  # Remove last row

return vertices
"""