from typing import List, Tuple

import commonroad_dc.pycrcc as pycrcc
from commonroad.visualization.mp_renderer import MPRenderer
import numpy as np
import triangle
from sympy import Point, Polygon
from sympy import Triangle


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

    #vertices.append((x_c, y_c))

    return vertices


def vertices_to_polygon(vertices: List[Tuple]) -> Polygon:
    points = [Point(vertex) for vertex in vertices]
    car_poly = Polygon(*points)

    return car_poly


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


def triangulate_car(vertices: List[Tuple]) -> List[Triangle]:
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
        car_mesh.append(Triangle(Point(v0[0], v0[1]),
                                 Point(v1[0], v1[1]),
                                 Point(v2[0], v2[1])))
    return car_mesh


def transform_mesh(car_mesh_l: List[Triangle], car: pycrcc.RectOBB) -> List[Triangle]:
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
        for vertex in triangle_l.vertices:
            x_g = (vertex[0] * cos - vertex[1] * sin) + x_c
            y_g = (vertex[0] * sin + vertex[1] * cos) + y_c
            vertices_g.append(Point(x_g, y_g))

        car_mesh_g.append(Triangle(*vertices_g))

    return car_mesh_g


def generate_mesh(car: pycrcc.RectOBB) -> List[Triangle]:
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

def rectobb_to_poly(car: pycrcc.RectOBB) -> Polygon:
    vertices_g = get_vertices_global(car)
    car = vertices_to_polygon(vertices_g)
    return car

def print_point_list(points: List[Point]):
    for point in points:
        print(f"({float(point[0])},{float(point[1])})")

def test_impact_location():
    """
    Test that prints location of impact when there is a collision
    """
    # Create two rectangles
    # car_a = pycrcc.RectOBB(1.0, 2.0, 1.2, 5.0, 9.0)  # green left
    car_a = pycrcc.RectOBB(1.0, 2.0, 1.2, 10.0, 9.0)  # green left
    car_b = pycrcc.RectOBB(1.0, 2.0, np.pi, 8.0, 10.0)  # red

    car_a_mesh = generate_mesh(car_a)
    car_b_mesh = generate_mesh(car_b)

    car_a = rectobb_to_poly(car_a)
    car_b = rectobb_to_poly(car_b)

    impact_loc = {0: "left", 1: "right", 2: "rear", 3: "front"}

    # Check collision between two cars
    collision = car_a.intersection(car_b)  # this implies car_a moving towards car_b
    if len(collision):
        print(f"Detected a collision")

        # If they a collides with b, check with which zone
        impact_zone = {}  # Dict[zone: area]
        for count, zone in enumerate(car_b_mesh):
            tmp = car_a.intersection(zone)
            tmp_size = len(tmp)
            # If a collides with zone
            if tmp_size:
                print(f"A collided with B-{impact_loc[count]}")
                # If polygon collision
                if tmp_size > 2:
                    impact_zone[count] = float(Polygon(*tmp).area)
                # If segment collision
                else:
                    impact_zone[count] = 0.0

    print('Collision between A and B: ', car_a.intersection(car_b))

class CollisionReport:
    impact_location: Mapping[ImpactLocation, pycrcc.Triangle]

    def __init__(self):
        self.impact_location = {}