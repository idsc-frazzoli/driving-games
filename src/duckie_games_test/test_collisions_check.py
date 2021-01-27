from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np
from decimal import Decimal as D
import os
from math import isclose


from duckie_games.rectangle import (
    ProjectedCar,
    Rectangle,
)
from duckie_games.collisions import IMPACT_RIGHT, IMPACT_LEFT, IMPACT_FRONT, IMPACT_BACK
from duckie_games.collisions_check import who_at_fault_line_of_sight, get_angle_of_collision, get_impact_location


module_path = os.path.dirname(__file__)


def test_collision_check_utils():
    """
    Tests the help functions visually
    """
    background_path = "out/map_drawing/4way/drawing.png"
    background_fp = os.path.join(
        module_path,
        background_path
    )
    x_back = 35
    y_back = 35

    rect_params = {
        "orientation": [
            [D(50), D(-20)],
            [D(10), D(9)],
            [D(0), D(180)],
            [D(-42), D(-90)]
        ],
        "translation": [
            [(D(25), D(4)), (D(22), D(4.3))],
            [(D(5), D(31.5)), (D(5), D(33.5))],
            [(D(6), D(5)), (D(9), D(4))],
            [(D(32), D(32)), (D(32), D(28))]
        ],
        "length": [
            [D(4.8), D(4.8)],
            [D(4.8), D(4.8)],
            [D(4.8), D(4.8)],
            [D(4.8), D(4.8)]
        ],
        "width": [
            [D(1.8), D(1.8)],
            [D(1.8), D(1.8)],
            [D(1.8), D(1.8)],
            [D(1.8), D(1.8)]
        ],
        "impact-location": [
            [IMPACT_LEFT, IMPACT_FRONT],
            [IMPACT_LEFT, IMPACT_RIGHT],
            [IMPACT_FRONT, IMPACT_FRONT],
            [IMPACT_RIGHT, IMPACT_BACK]
        ],
        "line_of_sight": [
            [False, True],
            [True, True],
            [True, True],
            [True, False]
        ],
        "collision-angle": [
            [D(-70), D(70)],
            [D(-1), D(1)],
            [D(180), D(180)],
            [D(-48), D(48)]
        ]
    }

    fig, ax = plt.subplots()
    ax.set_title("Collision Check Utils Test")

    try:
        img = imread(background_fp)
        ax.imshow(img, extent=[0, x_back, 0, y_back])
    except FileNotFoundError:
        ax.set_xlim(left=0, right=x_back)
        ax.set_ylim(bottom=0, top=y_back)

    for i in range(len(list(rect_params.values())[0])):

        orientation1 = rect_params["orientation"][i][0]
        translation1 = rect_params["translation"][i][0]
        height1 = rect_params["width"][i][0]
        width1 = rect_params["length"][i][0]

        orientation2 = rect_params["orientation"][i][1]
        translation2 = rect_params["translation"][i][1]
        height2 = rect_params["width"][i][1]
        width2 = rect_params["length"][i][1]

        center_pose1 = (translation1[0], translation1[1], orientation1)
        rect1 = Rectangle(
            center_pose=center_pose1,
            width=width1,
            height=height1
        )

        contour1 = rect1.closed_contour
        front_left1 = contour1[0]
        front_right1 = contour1[-2]
        front_center1 = front_left1 + (front_right1 - front_left1) / 2

        xy_front1 = np.array([front_left1, front_right1, front_center1]).T
        x_front1, y_front1 = xy_front1[0, :], xy_front1[1, :]


        proj_car1 = ProjectedCar(
            rectangle=rect1,
            front_right=front_right1,
            front_center=front_center1,
            front_left=front_left1
        )

        contour_np1 = np.array(contour1).T
        x_cont1, y_cont1 = contour_np1[0, :], contour_np1[1, :]

        center_pose2 = (translation2[0], translation2[1], orientation2)
        rect2 = Rectangle(
            center_pose=center_pose2,
            width=width2,
            height=height2
        )
        contour2 = rect2.closed_contour

        front_left2 = contour2[0]
        front_right2 = contour2[-2]
        front_center2 = front_left2 + (front_right2 - front_left2) / 2

        xy_front2 = np.array([front_left2, front_right2, front_center2]).T
        x_front2, y_front2 = xy_front2[0, :], xy_front2[1, :]

        proj_car2 = ProjectedCar(
            rectangle=rect2,
            front_right=front_right2,
            front_center=front_center2,
            front_left=front_left2
        )

        contour_np2 = np.array(contour2).T
        x_cont2, y_cont2 = contour_np2[0, :], contour_np2[1, :]

        ax.plot(x_cont1, y_cont1, linewidth=2)
        ax.plot(*translation1, 'x')
        ax.plot(x_front1, y_front1, 'x', linewidth=1.5)

        ax.plot(x_cont2, y_cont2, linewidth=2)
        ax.plot(*translation2, 'x')
        ax.plot(x_front2, y_front2, 'x', linewidth=1.5)

        ref_col_angle =  rect_params["collision-angle"][i]

        col_angle = get_angle_of_collision(a=proj_car1, b=proj_car2)

        assert all(map(isclose, ref_col_angle, col_angle)), (
            f"Collision angle function does not work.\n"
            f"ref {ref_col_angle} is not {col_angle}"
        )

        ref_line_sight = rect_params["line_of_sight"][i]
        line_sight = who_at_fault_line_of_sight(a=proj_car1, b=proj_car2)

        _isequal = lambda x, y: x == y

        assert all(map(_isequal, ref_line_sight, line_sight)), (
            f"Who at fault line of sight function does not work.\n"
            f"ref {ref_line_sight} is not {line_sight}"
        )

        ref_impact_location = rect_params["impact-location"][i]

        impact_location = get_impact_location(a=proj_car1, b=proj_car2)

        assert all(map(_isequal, ref_line_sight, line_sight)), (
            f"Impact location function does not work.\n"
            f"ref {ref_impact_location} is not {impact_location}"
        )
    try:
        fig.savefig("out/test_collision_check_utils.png")
    except FileNotFoundError:
        os.mkdir('out')
        fig.savefig("out/test_collision_check_utils.png")

    fig.tight_layout()
    fig.show()
    plt.close(fig=fig)