from typing import List

import commonroad_dc.pycrcc as pycrcc
from commonroad.visualization.mp_renderer import MPRenderer


def test_main():
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
    rnd.render(show=True, filename='hola.png')

    print('Collision between OBB and AABB: ', obb.collide(aabb))
    print('Collision between AABB and Circle: ', aabb.collide(circ))
    print('Collision between Circle and OBB:  ', circ.collide(obb))


def test_ugly_impact():
    # Create two rectangles
    car_a = pycrcc.RectOBB(1.0, 2.0, 0.3, 9.0, 8.0)  # green
    car_b = pycrcc.RectOBB(1.0, 2.0, 0.3, 8.0, 10.0)  # red

    rnd = MPRenderer(figsize=(10, 10))
    car_a.draw(rnd, draw_params={'facecolor': 'green'})
    car_b.draw(rnd, draw_params={'facecolor': 'red'})
    rnd.render(show=True, filename='hola.png')

    collision = car_a.collide(car_b)  # this implies car_a moving towards car_b
    if collision:
        print(f"Detected a collision")

        #if car_a.collide(car_b.front)

    print('Collision between car_a and car_b: ', car_a.collide(car_b))
