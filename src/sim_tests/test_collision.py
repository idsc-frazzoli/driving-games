import numpy as np
from geometry import SO2_from_angle
from matplotlib import pyplot as plt
from shapely.geometry import Polygon

from sim.collision import get_vehicle_mesh
from sim.models.vehicle_structures import VehicleGeometry


def test_rotation():
    delta = 0.8
    vec = np.array([0.58, 4.21])
    rot_delta = SO2_from_angle(delta)
    vec_rot = rot_delta @ vec
    vec2 = rot_delta.T @ vec_rot
    rot_mdelta = SO2_from_angle(-delta)
    vec3 = rot_mdelta @ vec_rot
    np.testing.assert_array_almost_equal(vec, vec2)
    np.testing.assert_array_almost_equal(vec, vec3)


def test_impact_location():
    """
    Test that prints location of impact when there is a collision
    """
    # Create two rectangles
    vg = VehicleGeometry.default_car()
    footprint = Polygon(vg.outline)
    impact_locations = get_vehicle_mesh(footprint)
    fig = plt.figure()

    for loc, triangle in impact_locations.items():
        plt.plot(*triangle.exterior.xy)
        xc, yc = triangle.centroid.coords[0]
        plt.text(xc, yc, f"{loc}", horizontalalignment="center", verticalalignment="center")

    fig.set_tight_layout(True)
    plt.axis('equal')
    plt.show()
