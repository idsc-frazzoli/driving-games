from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np
from decimal import Decimal as D
import os

from driving_games.structures import NO_LIGHTS

from duckie_games.rectangle import (
    Rectangle,
    get_resources_used,
    projected_car_from_along_lane,
)
from duckie_games.zoo import two_player_duckie_game_parameters_stretched
from duckie_games.utils import LaneSegmentHashable
from duckie_games.structures import DuckieState, DuckieGeometry

module_path = os.path.dirname(__file__)

def test_rectangle():
    background_path = "out/map_drawing/4way/drawing.png"
    x_back = 7
    y_back = 7

    orientation = D(50)
    translation = D(3), D(4)
    height = D(4)
    width = D(2)

    center_pose = (translation[0], translation[1], orientation)
    rect = Rectangle(
        center_pose=center_pose,
        width=width,
        height=height
    )
    contour = rect.closed_contour
    contour_np = np.array(contour).T
    x_cont, y_cont = contour_np[0, :], contour_np[1, :]

    point_inside = rect.get_points_inside()
    point_inside_np = np.array(point_inside).T
    x_pin, y_pin = point_inside_np[0, :], point_inside_np[1, :]

    background_fp = os.path.join(
        module_path,
        background_path
    )
    fig, ax = plt.subplots()
    ax.set_title("Rectangle Test")
    img = imread(background_fp)
    ax.imshow(img, extent=[0, x_back, 0, y_back])
    ax.plot(x_cont, y_cont, linewidth=2)
    ax.plot(*translation, 'x')
    ax.plot(x_pin, y_pin, 'x')
    fig.savefig("out/rectangle_test.png")
    fig.tight_layout()
    fig.show()
    plt.close(fig=fig)


def test_resources():
    background_path = "out/map_drawing/4way/drawing.png"
    x_back = 7
    y_back = 7

    duckie_name = two_player_duckie_game_parameters_stretched.player_names[1]
    duckie_map = two_player_duckie_game_parameters_stretched.duckie_map
    lane = two_player_duckie_game_parameters_stretched.lanes[duckie_name]
    ref = two_player_duckie_game_parameters_stretched.refs[duckie_name]

    shared_resources_ds = D(1.5)
    length = D(5)
    width = D(1.8)
    along_lane = D(12)
    speed = D(1)

    duckie_x = DuckieState(
        ref=ref,
        x=along_lane,
        lane=LaneSegmentHashable.initializor(lane),
        wait=D(0),
        v=speed,
        light=NO_LIGHTS
    )

    duck_g = DuckieGeometry(
        mass=D(1000),
        length = length,
        width = width,
        color=(1, 0, 0),
        height=D(2)
    )


    resources = get_resources_used(vs=duckie_x, vg=duck_g, ds=shared_resources_ds)

    fig, ax = plt.subplots()
    ax.set_title("Resources Test")
    background_fp = os.path.join(
        module_path,
        background_path
    )

    tile_size = duckie_map.tile_size
    H = duckie_map['tilemap'].H
    W = duckie_map['tilemap'].W
    x_size = tile_size * W
    y_size = tile_size * H
    img = imread(background_fp)
    ax.imshow(img, extent=[0, x_size, 0, y_size])

    for rectangle in resources:
        countour_points = np.array(rectangle.closed_contour).T
        x, y = countour_points[0, :], countour_points[1, :]
        ax.plot(x, y, linewidth=0.5)

    proj_car = projected_car_from_along_lane(lane=lane, along_lane=along_lane, vg=duck_g)
    car_contour = proj_car.rectangle.closed_contour
    contour_np = np.array(car_contour).T
    x_cont, y_cont = contour_np[0, :], contour_np[1, :]
    ax.plot(x_cont, y_cont, linewidth=2)

    fig.savefig("out/resources_test.png")
    fig.tight_layout()
    fig.show()
    plt.close(fig=fig)