import commonroad_dc.pycrcc as pycrcc
from commonroad.visualization.mp_renderer import MPRenderer

from sim.collision import get_rectangle_mesh, get_impact_locations


def test_commonroad_dc():
    aabb = pycrcc.RectAABB(2.0, 3.0, 3.0, 2.0)

    # Oriented rectangle with width/2, height/2, orientation, x-position , y-position
    obb = pycrcc.RectOBB(1.0, 2.0, 0.3, 8.0, 10.0)

    # Circle with radius, x-position , y-position
    circ = pycrcc.Circle(2.5, 6.0, 7.0)

    # Triangle with vertices (x1, y1), (x2, y2), and (x3, y3)
    tri = pycrcc.Triangle(0.0, 0.0, 4.0, 0.0, 2.0, 2.0)

    rnd = MPRenderer(figsize=(10, 10))
    aabb.draw(rnd, draw_params={'facecolor': 'green'})
    obb.draw(rnd, draw_params={'facecolor': 'red'})
    circ.draw(rnd, draw_params={'facecolor': 'yellow'})
    tri.draw(rnd, draw_params={'facecolor': 'blue'})
    rnd.render(show=True)

    print('Collision between OBB and AABB: ', obb.collide(aabb))
    print('Collision between AABB and Circle: ', aabb.collide(circ))
    print('Collision between Circle and OBB:  ', circ.collide(obb))


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

    col_repA = get_impact_locations(car_a, car_b)
    col_repB = get_impact_locations(car_b, car_a)
    print(f'Collision report A: {col_repA}')
    print(f'Collision report B: {col_repB}')
