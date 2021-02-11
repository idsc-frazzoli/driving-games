import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import imread
from decimal import Decimal as D

from driving_games.structures import NO_LIGHTS
from world.map_loading import load_driving_game_map, map_directory
from world.utils import LaneSegmentHashable

from duckie_games.zoo import two_player_4way
from duckie_games.shared_resources import DrivingGameGridMap, get_resources_used
from duckie_games.rectangle import projected_car_from_along_lane, Rectangle
from duckie_games.structures import DuckieState, DuckieGeometry


def test_visualize_resource_grid():
    """
    Tests the resource grid visually
    """
    map_name = "4way"
    driving_game_map = load_driving_game_map(map_name)
    resource_cell_size = D(1)
    driving_game_grid_map = DrivingGameGridMap.initializor(m=driving_game_map, cell_size=resource_cell_size)

    background_path = os.path.join(map_directory, f"{map_name}.png")
    tile_size = driving_game_grid_map.tile_size
    H = driving_game_grid_map['tilemap'].H
    W = driving_game_grid_map['tilemap'].W
    x_back = tile_size * W
    y_back = tile_size * H

    fig, ax = plt.subplots()
    ax.set_title("Resource Grid Test")

    try:
        img = imread(background_path)
        ax.imshow(img, extent=[0, x_back, 0, y_back])
    except FileNotFoundError:
        ax.set_xlim(left=0, right=x_back)
        ax.set_ylim(bottom=0, top=y_back)

    for coords in driving_game_grid_map.resources.values():
        ax.plot(*coords, 'x', color='red')

    try:
        fig.savefig("out/test_visualize_resource_grid.png")
    except FileNotFoundError:
        os.mkdir('out')
        fig.savefig("out/test_visualize_resource_grid.png")

    fig.tight_layout()
    fig.show()
    plt.close(fig=fig)


def test_resources_visual():
    """
    Tests the get resources function visually
    """
    map_name = two_player_4way.map_name
    driving_game_map = load_driving_game_map(map_name)
    resource_cell_size = D(0.5)
    driving_game_grid_map = DrivingGameGridMap.initializor(m=driving_game_map, cell_size=resource_cell_size)

    background_path = os.path.join(map_directory, f"{map_name}.png")
    tile_size = driving_game_grid_map.tile_size
    H = driving_game_grid_map['tilemap'].H
    W = driving_game_grid_map['tilemap'].W
    x_back = tile_size * W
    y_back = tile_size * H

    duckie_name = two_player_4way.player_names[0]
    lane = two_player_4way.lanes[duckie_name]
    ref = two_player_4way.refs[duckie_name]

    length = D(5)
    width = D(1.8)
    along_lane = D(10)
    speed = D(2)

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
        length=length,
        width=width,
        color=(1, 0, 0),
        height=D(2)
    )

    resources = get_resources_used(vs=duckie_x, vg=duck_g, m=driving_game_grid_map)

    fig, ax = plt.subplots()
    ax.set_title("Resources Test")

    try:
        img = imread(background_path)
        ax.imshow(img, extent=[0, x_back, 0, y_back])
    except FileNotFoundError:
        ax.set_xlim(left=0, right=x_back)
        ax.set_ylim(bottom=0, top=y_back)

    shared_resources_ds = driving_game_grid_map.cell_size
    for _id in resources:
        center_x, center_y = driving_game_grid_map.resources[_id]
        rec = Rectangle(center_pose=(center_x, center_y, D(0)), width=shared_resources_ds, height=shared_resources_ds)
        countour_points = np.array(rec.closed_contour).T
        x, y = countour_points[0, :], countour_points[1, :]
        ax.fill(x, y, linewidth=0.2, color=duck_g.color, alpha=0.5)

    proj_car = projected_car_from_along_lane(lane=lane, along_lane=along_lane, vg=duck_g)

    car_contour = proj_car.rectangle.closed_contour
    contour_np = np.array(car_contour).T
    x_cont, y_cont = contour_np[0, :], contour_np[1, :]
    ax.plot(x_cont, y_cont, color=duck_g.color, linewidth=2)

    front_left = proj_car.front_left
    front_right = proj_car.front_right
    front_center = proj_car.front_center

    xy_front = np.array([front_left, front_right, front_center]).T
    x_front, y_front = xy_front[0, :], xy_front[1, :]
    ax.plot(x_front, y_front, 'x', linewidth=1.5)

    try:
        fig.savefig("out/test_resources.png")
    except FileNotFoundError:
        os.mkdir('out')
        fig.savefig("out/test_resources.png")

    fig.tight_layout()
    fig.show()
    plt.close(fig=fig)
