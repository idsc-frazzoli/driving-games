from frozendict import frozendict

from toy_games.n_player_toy_structures import ToyCarMap, ToyLane
from toy_games.n_player_toy_game import ToyGameParams

__all__ = [
    "toy_params_x",
    "toy_params_star",
    "toy_params_x_with_base",
    "toy_params_indep_lanes",
    "toy_params_one_indep_lane",
    "toy_params_two_indep_games",
    "toy_params_two_x_joint",
    "toy_params_two_x_crossed"
]

max_wait = 2
"""This param is applied to all maps exept the 5 player one"""


toy_lane_x_1 = ToyLane(
    control_points=frozendict({
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        4: 5,
    })
)

toy_lane_x_2 = ToyLane(
    control_points=frozendict({
        0: 6,
        1: 7,
        2: 8,
        3: 4,
        4: 9
    })
)

toy_map_x = ToyCarMap(
    lanes=[
        toy_lane_x_1,
        toy_lane_x_2,
    ]
)
toy_params_x = ToyGameParams(
    params_name="2_player_x",
    toy_game_map=toy_map_x,
    max_wait=max_wait
)
""" 
This map contains 2-lanes. Collision happens at point (4):

           (6) x           x (1)
                \         /
             (7) x       x (2)
                  \     /
               (8) x   x (3)
                    \ /
                     x (4)
                    / \ 
               (5) x   x (9)

"""


toy_lane_star_1 = ToyLane(
    control_points=frozendict({
        0: 1,
        1: 7,
        2: 4
    })
)

toy_lane_star_2 = ToyLane(
    control_points=frozendict({
        0: 2,
        1: 7,
        2: 5,
        3: 29
    })
)

toy_lane_star_3 = ToyLane(
    control_points=frozendict({
        0: 3,
        1: 7,
        2: 6,
        3: 22
    })
)

toy_map_star = ToyCarMap(
    lanes=[
        toy_lane_star_1,
        toy_lane_star_2,
        toy_lane_star_3
    ]
)
toy_params_star = ToyGameParams(
    params_name="3_player_star",
    toy_game_map=toy_map_star,
    max_wait=max_wait
)
""" 
This map contains 3-lanes. Collision happens at point (7):
                 (1) 
           (2) x  x  x (6)
                \ | /
                 \|/
                  x (7)
                 /|\ 
                / | \ 
           (3) x  x  x (5)
                 (4)
"""

toy_lane_x_with_base_1 = ToyLane(
    control_points=frozendict({
        0: 1,
        1: 6,
        2: 3
    })
)

toy_lane_x_with_base_2 = ToyLane(
    control_points=frozendict({
        0: 2,
        1: 6,
        2: 5
    })
)

toy_lane_x_with_base_3 = ToyLane(
    control_points=frozendict({
        0: 4,
        1: 3,
        2: 5,
        3: 7
    })
)

toy_map_x_with_base = ToyCarMap(
    lanes=[
        toy_lane_x_with_base_1,
        toy_lane_x_with_base_2,
        toy_lane_x_with_base_3
    ]
)
toy_params_x_with_base = ToyGameParams(
    params_name="3_player_x_with_base",
    toy_game_map=toy_map_x_with_base,
    max_wait=max_wait
)
"""
This map contains 3 lanes. Collisions happen at (3), (5) and (6)

           (2) x     x (1)
                \   /
                 \ /
                  x (6)
                 / \ 
             (3)/   \ (5)
       (4) x---x-----x----x (7)

"""


toy_lane_indep_lanes_1 = ToyLane(
    control_points=frozendict({
        0: 1,
        1: 2,
        2: 3,
        3: 4
    })
)

toy_lane_indep_lanes_2 = ToyLane(
    control_points=frozendict({
        0: 5,
        1: 6,
        2: 7,
        3: 8
    })
)

toy_lane_indep_lanes_3 = ToyLane(
    control_points=frozendict({
        0: 9,
        1: 10,
        2: 11,
        3: 12
    })
)

toy_map_indep_lanes = ToyCarMap(
    lanes=[
        toy_lane_indep_lanes_1,
        toy_lane_indep_lanes_2,
        toy_lane_indep_lanes_3
    ]
)

toy_params_indep_lanes = ToyGameParams(
    params_name="3_player_indep_lanes",
    toy_game_map=toy_map_indep_lanes,
    max_wait=max_wait
)
"""
This map contains 3 independent lanes. No collision happened

                (2)    (3)
       (1) x-----x-----x-----x (4)
                (6)   (7)
       (5) x-----x-----x-----x (8)
                 (10)  (11)
       (9) x-----x-----x-----x (12)

"""


toy_lane_one_indep_lane_1 = ToyLane(
    control_points=frozendict({
        0: 1,
        1: 5,
        2: 3
    })
)

toy_lane_one_indep_lane_2 = ToyLane(
    control_points=frozendict({
        0: 2,
        1: 5,
        2: 4
    })
)

toy_lane_one_indep_lane_3 = ToyLane(
    control_points=frozendict({
        0: 6,
        1: 7,
        2: 8
    })
)

toy_map_one_indep_lane = ToyCarMap(
    lanes=[
        toy_lane_one_indep_lane_1,
        toy_lane_one_indep_lane_2,
        toy_lane_one_indep_lane_3
    ]
)

toy_params_one_indep_lane = ToyGameParams(
    params_name="3_player_one_indep_lane",
    toy_game_map=toy_map_one_indep_lane,
    max_wait=max_wait
)
"""
This map contains 1 independent lane. Collision at (5)

               x (1)
          (5) /   
   (2) x-----x-----x (4)
            /  
           x (3)
  (6) x-----x-----x (8)
           (7)
"""


toy_lane_two_indep_games_1 = ToyLane(
    control_points=frozendict({
        0: 1,
        1: 5,
        2: 7,
        3: 3
    })
)

toy_lane_two_indep_games_2 = ToyLane(
    control_points=frozendict({
        0: 2,
        1: 6,
        2: 7,
        3: 4
    })
)

toy_lane_two_indep_games_3 = ToyLane(
    control_points=frozendict({
        0: 8,
        1: 12,
        2: 14,
        3: 10
    })
)

toy_lane_two_indep_games_4 = ToyLane(
    control_points=frozendict({
        0: 9,
        1: 13,
        2: 14,
        3: 11
    })
)

toy_map_two_indep_games = ToyCarMap(
    lanes=[
        toy_lane_two_indep_games_1,
        toy_lane_two_indep_games_2,
        toy_lane_two_indep_games_3,
        toy_lane_two_indep_games_4
    ]
)


toy_params_two_indep_games = ToyGameParams(
    params_name="4_player_two_indep_games",
    toy_game_map=toy_map_two_indep_games,
    max_wait=max_wait
)
"""
This map contains 2 independent games. Collision at (7) and (13)

    (2) x       x (1)   (9) x       x (8) 
         \     /             \     /
      (6) x   x (5)      (13) x   x (12)
           \ /                 \ /
            x (7)               x (14)
           / \                 / \ 
          /   \               /   \ 
     (3) x     x (4)     (10) x     x (11)
"""


toy_lane_two_x_joint_1 = ToyLane(
    control_points=frozendict({
        0: 1,
        1: 5,
        2: 3
    })
)

toy_lane_two_x_joint_2 = ToyLane(
    control_points=frozendict({
        0: 2,
        1: 5,
        2: 4
    })
)

toy_lane_two_x_joint_3 = ToyLane(
    control_points=frozendict({
        0: 6,
        1: 9,
        2: 4
    })
)

toy_lane_two_x_joint_4 = ToyLane(
    control_points=frozendict({
        0: 7,
        1: 9,
        2: 8
    })
)

toy_map_two_x_joint = ToyCarMap(
    lanes=[
        toy_lane_two_x_joint_1,
        toy_lane_two_x_joint_2,
        toy_lane_two_x_joint_3,
        toy_lane_two_x_joint_4
    ]
)

toy_params_two_x_joint = ToyGameParams(
    params_name="4_player_two_x_joint",
    toy_game_map=toy_map_two_x_joint,
    max_wait=max_wait
)
"""
This map contains a joint 4 player game. Collision at (4), (5) and (9)
             (1) (7)
    (2) x     x  x     x (6) 
         \   /    \   /
          \ /      \ /
           x (5)    x (9)
          / \      / \ 
         /   \    /   \ 
    (3) x     \  /     x (8)
               x
              (4)
"""


max_wait = 6

toy_lane_two_x_crossed_1 = ToyLane(
    control_points=frozendict({
        0: 1,
        1: 5,
        2: 3
    })
)

toy_lane_two_x_crossed_2 = ToyLane(
    control_points=frozendict({
        0: 2,
        1: 5,
        2: 4
    })
)

toy_lane_two_x_crossed_3 = ToyLane(
    control_points=frozendict({
        0: 6,
        1: 9,
        2: 4
    })
)

toy_lane_two_x_crossed_4 = ToyLane(
    control_points=frozendict({
        0: 7,
        1: 9,
        2: 8
    })
)

toy_lane_two_x_crossed_5 = ToyLane(
    control_points=frozendict({
        0: 10,
        1: 5,
        2: 9,
        3: 11
    })
)

toy_map_two_x_crossed = ToyCarMap(
    lanes=[
        toy_lane_two_x_crossed_1,
        toy_lane_two_x_crossed_2,
        toy_lane_two_x_crossed_3,
        toy_lane_two_x_crossed_4,
        toy_lane_two_x_crossed_5,
    ]
)

toy_params_two_x_crossed = ToyGameParams(
    params_name="5_player_two_x_crossed",
    toy_game_map=toy_map_two_x_crossed,
    max_wait=max_wait
)
"""
This map contains a joint 5 player game. Collision at (4), (5) and (9)
              (1) (7)
     (2) x     x  x     x (6) 
          \   /    \   /
           \ /      \ /
(10) x------x-(5)----x-(9)-----x (11)
           / \      / \ 
          /   \    /   \ 
     (3) x     \  /     x (8)
                x
               (4)
"""
