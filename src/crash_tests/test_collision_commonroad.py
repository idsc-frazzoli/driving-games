from typing import List, Tuple

import commonroad_dc.pycrcc as pycrcc
from commonroad.visualization.mp_renderer import MPRenderer
import numpy as np
import triangle


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


'''
class VehicleZones:
    front: pycrcc.Triangle
    left: pycrcc.Triangle
    right: pycrcc.Triangle
    rear: pycrcc.Triangle

    def __init__(self, car: pycrcc.RectOBB):
        # todo -> get RectOBB vertex and divide in 4 triangles

        # self.front = pycrcc.Triangle(car.getAABB().min_x,)
        # self.left
        # self.right
        # self.rear
        pass
'''


def get_vertices(car: pycrcc.RectOBB) -> List[Tuple]:
    """
    This returns all the vertices of a rectangle in the global reference frame
    :param car:
    :return:
    """
    x_c = car.center()[0]
    y_c = car.center()[1]
    x_l = [car.r_x(), -car.r_x(), -car.r_x(), car.r_x()]
    y_l = [car.r_y(), car.r_y(), -car.r_y(), -car.r_y()]
    theta = car.orientation()

    vertices = []

    for x, y in zip(x_l, y_l):
        x_g = (x * np.cos(theta) - y * np.sin(theta)) + x_c
        y_g = (x * np.sin(theta) + y * np.cos(theta)) + y_c
        vertices.append((x_g, y_g))

    vertices.append((x_c, y_c))

    return vertices


def triangulate_car(vertices: List[Tuple]) -> List[pycrcc.Triangle]:
    # triangulate the polygon
    number_of_vertices = len(vertices)
    segments = list(zip(range(0, number_of_vertices - 1), range(1, number_of_vertices)))
    segments.append((0, number_of_vertices - 1))
    triangles = triangle.triangulate({'vertices': vertices, 'segments': segments}, opts='pqS2.4')
    # convert all triangles to pycrcc.Triangle
    car_triangulated = list()
    for t in triangles['triangles']:
        v0 = triangles['vertices'][t[0]]
        v1 = triangles['vertices'][t[1]]
        v2 = triangles['vertices'][t[2]]
        car_triangulated.append(pycrcc.Triangle(v0[0], v0[1],
                                    v1[0], v1[1],
                                    v2[0], v2[1]))
    return car_triangulated

# fixme: there is one triangle missing!! Fix this

def test_ugly_impact():
    # Create two rectangles
    car_a = pycrcc.RectOBB(1.0, 2.0, 0.3, 9.0, 8.0)  # green
    car_b = pycrcc.RectOBB(1.0, 2.0, 0.3, 8.0, 10.0)  # red

    rnd = MPRenderer(figsize=(10, 10))
    car_a.draw(rnd, draw_params={'facecolor': 'green'})
    car_b.draw(rnd, draw_params={'facecolor': 'red'})
    rnd.render(show=True)

    vertices_a = get_vertices(car_a)
    vertices_b = get_vertices(car_b)

    car_a_t = triangulate_car(vertices_a)
    car_b_t = triangulate_car(vertices_b)

    rnd2 = MPRenderer(figsize=(10, 10))
    rnd2.draw_list(car_a_t, draw_params={'facecolor': 'orange', 'draw_mesh': False})
    rnd2.draw_list(car_b_t, draw_params={'facecolor': 'orange', 'draw_mesh': False})
    rnd2.render(show=True)

    collision = car_a.collide(car_b)  # this implies car_a moving towards car_b
    if collision:
        print(f"Detected a collision")

        # if car_a.collide(car_b.front)

    print('Collision between car_a and car_b: ', car_a.collide(car_b))
