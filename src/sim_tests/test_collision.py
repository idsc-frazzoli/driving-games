import commonroad_dc.pycrcc as pycrcc
import numpy as np
from commonroad.visualization.mp_renderer import MPRenderer
from geometry import SO2_from_angle

from sim.collision import get_rectangle_mesh, impact_locations_from_polygons


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
    # car_a = pycrcc.RectOBB(1.0, 2.0, 1.2, 5.0, 9.0)  # green left
    car_a = pycrcc.RectOBB(1.0, 2.0, 1.2, 10.0, 9.0)  # green left
    car_b = pycrcc.RectOBB(1.0, 2.0, 0, 8.0, 10.0)  # red

    print(car_a.r_x())
    print(car_a.r_y())

    car_a_mesh = list(get_rectangle_mesh(car_a).values())
    car_b_mesh = list(get_rectangle_mesh(car_b).values())

    rnd2 = MPRenderer(figsize=(10, 10))
    rnd2.draw_list(car_a_mesh, draw_params={'facecolor': 'blue', 'draw_mesh': False})
    rnd2.draw_list(car_b_mesh, draw_params={'facecolor': 'orange', 'draw_mesh': False})
    rnd2.render(show=True)

    col_repA = impact_locations_from_polygons(car_a, car_b)
    col_repB = impact_locations_from_polygons(car_b, car_a)
    print(f'Collision report A: {col_repA}')
    print(f'Collision report B: {col_repB}')
