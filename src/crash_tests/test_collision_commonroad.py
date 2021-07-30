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
    rnd.render(show=True)

    print('Collision between OBB and AABB: ', obb.collide(aabb))
    print('Collision between AABB and Circle: ', aabb.collide(circ))
    print('Collision between Circle and OBB:  ', circ.collide(obb))


def get_vertices_global(car: pycrcc.RectOBB) -> List[Tuple]:
    """
    This returns all the vertices of a rectangle in the global reference frame of the map (a bit useless for now)
    :param car:
    :return:
    """
    x_c = car.center()[0]
    y_c = car.center()[1]

    theta = car.orientation()

    x_l = [car.r_x(), -car.r_x(), -car.r_x(), car.r_x()]
    y_l = [car.r_y(), car.r_y(), -car.r_y(), -car.r_y()]

    vertices = []

    for x, y in zip(x_l, y_l):
        x_g = (x * np.cos(theta) - y * np.sin(theta)) + x_c
        y_g = (x * np.sin(theta) + y * np.cos(theta)) + y_c
        vertices.append((x_g, y_g))

    vertices.append((x_c, y_c))

    return vertices


def get_vertices_local(car: pycrcc.RectOBB) -> List[Tuple]:
    """
    This returns all the vertices of a rectangle in the local reference frame of the car.
    The list starts on the top-right corner and moves anti-clockwise
    :param car:
    :return:
    """
    x_l = [car.r_x(), -car.r_x(), -car.r_x(), car.r_x(), 0]
    y_l = [car.r_y(), car.r_y(), -car.r_y(), -car.r_y(), 0]

    vertices = [(x, y) for x, y in zip(x_l, y_l)]
    return vertices


def triangulate_car(vertices: List[Tuple]) -> List[pycrcc.Triangle]:
    """
    This triangulates the car in 4 triangles based on its diagonals and returns them in a list
    :param vertices:
    :return:
    """
    # triangulate the polygon
    number_of_vertices = len(vertices)
    segments = list(zip(range(0, number_of_vertices - 1), range(1, number_of_vertices)))
    segments.append((0, number_of_vertices - 1))
    triangles = triangle.triangulate({'vertices': vertices, 'segments': segments})
    # convert all triangles to pycrcc.Triangle
    car_mesh = list()
    for t in triangles['triangles']:
        v0 = triangles['vertices'][t[0]]
        v1 = triangles['vertices'][t[1]]
        v2 = triangles['vertices'][t[2]]
        car_mesh.append(pycrcc.Triangle(v0[0], v0[1],
                                        v1[0], v1[1],
                                        v2[0], v2[1]))
    return car_mesh


def transform_mesh(car_mesh_l: List[pycrcc.Triangle], car: pycrcc.RectOBB) -> List[pycrcc.Triangle]:
    """
    This rotates and translates a list of triangles in the local RF of the car to the global RF of the map
    :param car_mesh_l:
    :param car:
    :return:
    """
    x_c = car.center()[0]
    y_c = car.center()[1]
    theta = car.orientation()
    cos = np.cos(theta)
    sin = np.sin(theta)

    car_mesh_g = list()
    for count, triangle_l in enumerate(car_mesh_l):

        vertices_g = list()
        for vertex in triangle_l.vertices():
            x_g = (vertex[0] * cos - vertex[1] * sin) + x_c
            y_g = (vertex[0] * sin + vertex[1] * cos) + y_c
            vertices_g.extend([x_g, y_g])

        car_mesh_g.append(pycrcc.Triangle(*vertices_g))

    return car_mesh_g


def generate_mesh(car: pycrcc.RectOBB) -> List[pycrcc.Triangle]:
    """
    This function:
        1) Gets the vertices of the car in the local RF
        2) Triangulates the car
        3) Transforms the triangles mesh to global RF and returns this list of global triangles
    :param car:
    :return:
    """
    vertices = get_vertices_local(car)
    car_mesh_local = triangulate_car(vertices)
    car_mesh_global = transform_mesh(car_mesh_local, car)
    return car_mesh_global


def test_impact_location():
    """
    Test that prints location of impact when there is a collision
    """
    # Create two rectangles
    # car_a = pycrcc.RectOBB(1.0, 2.0, 1.2, 5.0, 9.0)  # green left
    car_a = pycrcc.RectOBB(1.0, 2.0, 1.2, 10.0, 9.0)  # green left
    car_b = pycrcc.RectOBB(1.0, 2.0, 0, 8.0, 10.0)  # red

    car_a_mesh = generate_mesh(car_a)
    car_b_mesh = generate_mesh(car_b)

    rnd2 = MPRenderer(figsize=(10, 10))
    rnd2.draw_list(car_b_mesh, draw_params={'facecolor': 'orange', 'draw_mesh': False})
    rnd2.draw_list(car_a_mesh, draw_params={'facecolor': 'orange', 'draw_mesh': False})
    rnd2.render(show=True)

    impact_loc = {0: "left", 1: "right", 2: "rear", 3: "front"}

    collision = car_a.collide(car_b)  # this implies car_a moving towards car_b
    if collision:
        print(f"Detected a collision")

        count = 0
        for area in car_b_mesh:
            tmp = car_a.collide(area)
            if tmp:
                print(f"A collided with B-{impact_loc[count]}")

            count += 1

            # if car_a.collide(car_b.front)

    print('Collision between A and B: ', car_a.collide(car_b))
