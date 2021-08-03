# import Point, Polygon
import numpy as np
from sympy import Point, Polygon
from sympy import Triangle
import commonroad_dc.pycrcc as pycrcc
from commonroad.visualization.mp_renderer import MPRenderer
from shapely.ops import nearest_points
from shapely.geometry import Point, Polygon

def test_polygons():
    '''
    Test interesting library sympy! (Better than pycrcc)
    Allows to easily compute many metrics from any polygon
    '''
    # creating points using Point()
    p1, p2, p3 = map(Point, [(0, 0), (1, 0),  (5, 1)])
    #p1, p2, p3 = map(Point, [(10, 10), (11, 11), (12, 12)])
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


def test_numpy_to_listtuple():
    arr = [[1, 1], [2, 2]]
    lst = tuple([tuple(col) for col in zip(*arr)])
    print(lst)


def test_tuples():
    a = ((0, 0), (1, 1))
    print(a)
    b = ((2, 2),)
    print(b)
    b = a+b
    print(b)
    a = a + (a[0],)
    print(a)