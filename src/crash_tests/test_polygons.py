# import Point, Polygon
import commonroad_dc.pycrcc as pycrcc
import contracts
import numpy as np
from commonroad.visualization.mp_renderer import MPRenderer
from geometry import SO2_from_angle
from matplotlib import pyplot as plt
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

from sim.collision_utils import _find_intersection_points


def test_polygons():
    '''
    Test interesting library sympy! (Better than pycrcc)
    Allows to easily compute many metrics from any polygon
    '''
    # creating points using Point()
    p1, p2, p3 = map(Point, [(0, 0), (1, 0), (5, 1)])
    # p1, p2, p3 = map(Point, [(10, 10), (11, 11), (12, 12)])
    p4, p5, p6 = map(Point, [(3, 2), (1, -1), (0, 2)])

    # creating polygons using Polygon()
    poly1 = Polygon(p1, p2, p3)
    poly2 = Polygon(p4, p5, p6)

    triangle1 = Triangle(p1, p2, p3)
    triangle2 = Triangle(p4, p5, p6)

    is_intersection_triangle = triangle1.intersection(triangle2)
    print(is_intersection_triangle)

    if len(is_intersection_triangle):
        intersection_area = Polygon(*is_intersection_triangle).area
        print(float(intersection_area))

    # Triangle with vertices (x1, y1), (x2, y2), and (x3, y3)
    tri1 = pycrcc.Triangle(0, 0, 1, 0, 5, 1)
    tri2 = pycrcc.Triangle(3, 2, 1, -1, 0, 2)

    rnd = MPRenderer(figsize=(10, 10))
    tri1.draw(rnd, draw_params={'facecolor': 'blue'})
    tri2.draw(rnd, draw_params={'facecolor': 'green'})
    rnd.render(show=True)

    # using intersection()
    isIntersection_poly = poly1.intersection(poly2)


def test_shapely_nearest_point():
    point = Point(0, 0)
    square = Polygon(((0, 2), (1, 2), (1, 3), (0, 3), (0, 2)))
    nearest = nearest_points(point, square)
    print(f'x: {nearest[1].x}')
    print(f'y: {nearest[1].y}')
    a = [o.wkt for o in nearest]
    print(a)


def test_shapely_collision():
    a = Polygon(((0, 0), (2, 0), (2, 2), (0, 2), (0, 0)))
    b = Polygon(((2, 1), (3, 0), (5, 1), (3, 2), (2, 1)))
    c = Polygon(((1, 1), (2, 0), (4, 1), (2, 2), (1, 1)))
    d = Polygon(((1, -0.5), (2, -1), (4, 1), (2.5, 1.5), (1, -0.5)))
    plt.plot(*a.exterior.xy, "b")
    plt.plot(*b.exterior.xy, "r")
    plt.plot(*c.exterior.xy, "g")
    plt.plot(*d.exterior.xy, "y")
    plt.show()
    intersection = a.intersection(b)
    print(a.touches(b))  # True if they touch just on one Point
    print(a.intersects(b))
    print(intersection.coords)  # Here intersection is a Point

    intersection = a.intersection(c)
    print(a.touches(c))
    print(a.intersects(c))
    print(_find_intersection_points(a, c))
    print(_find_intersection_points(a, d))  # Here intersection is a Polygon


def test():
    import math
    from time import perf_counter_ns
    contracts.disable_all()
    theta = 0.1
    vx, vy = (2, 4)

    t1_start = perf_counter_ns()
    costh = math.cos(theta)
    sinth = math.sin(theta)
    xdot = vx * costh - vy * sinth
    ydot = vx * sinth + vy * costh
    t1_stop = perf_counter_ns()
    print(f"Elapsed time {t1_stop - t1_start} , xdot: {xdot} , ydot: {ydot}")

    t2_start = perf_counter_ns()
    rot = SO2_from_angle(theta)
    vel = np.array([vx, vy])
    xydot = rot@vel
    t2_stop = perf_counter_ns()
    print(f"Elapsed time {t2_stop - t2_start} , xdot: {xydot[0]} , ydot: {xydot[1]}")
