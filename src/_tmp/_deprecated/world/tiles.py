# coding=utf-8
import yaml

from duckietown_serialization_ds1 import Serializable

from duckietown_world.utils.memoizing import memoized_reset

# language=yaml
data = """

# lane = 22cm
# tile = 58.5
# lane_rel = 22/58.5
go_right: &go_right
    ~LaneSegment:
      width: &width 0.376
      control_points: 
        - ~SE2Transform:
            p: [-0.50, -0.22]
            theta_deg: 0
        - ~SE2Transform:
            p: [-0.30, -0.30]
            theta_deg: -45
        - ~SE2Transform:
            p: [-0.22, -0.50]
            theta_deg: -90
            
go_straight: &go_straight
    ~LaneSegment:
      width: *width
      control_points: 
        - ~SE2Transform:
            p: [-0.5, -0.22]
            theta_deg: 0
        - ~SE2Transform:
            p: [+0.5, -0.22]
            theta_deg: 0

go_left: &go_left
    ~LaneSegment:
      width: *width
      control_points: 
        - ~SE2Transform:
            p: [-0.50, -0.22]
            theta_deg: 0 

        - ~SE2Transform:
            p: [0.0, 0.0]
            theta_deg: 45

        - ~SE2Transform:
            p: [+0.22, +0.50]
            theta_deg: 90

straight:
    ~PlacedObject:
        children:
            lane1: *go_straight
            lane2: *go_straight
        spatial_relations:
            lane1: {~SE2Transform:}
            lane2: {~SE2Transform: {theta_deg: 180}}

curve_right: &curve_right
    ~PlacedObject:
        children:
            lane1: *go_right
            lane2: *go_left
        spatial_relations:
            lane1: {~SE2Transform:}
            lane2: {~SE2Transform: {theta_deg: 90}}


curve_left: &curve_left
    ~PlacedObject:
        children:
            curve: *curve_right
        spatial_relations: 
            curve: {~SE2Transform: {theta_deg: 270}}

1way: &1way
     ~PlacedObject:
        children:
            go_right: *go_right
            go_left: *go_left
            go_straight: *go_straight

4way:
    ~PlacedObject:
        children:
            a: *1way
            b: *1way
            c: *1way
            d: *1way
        spatial_relations:
            a: {~SE2Transform: {theta_deg: 0}}
            b: {~SE2Transform: {theta_deg: 90}}
            c: {~SE2Transform: {theta_deg: 180}}
            d: {~SE2Transform: {theta_deg: 270}}
            
#    |    |        
# ---      ----
#  
# -----------

3way_left: &3way_left
    ~PlacedObject:
        children:
            west_go_straight: *go_straight
            west_go_left: *go_left
            north_go_left: *go_left
            north_go_right: *go_right
            east_go_right: *go_right
            east_go_straight: *go_straight
        spatial_relations:
            west_go_straight: {~SE2Transform: {theta_deg: 0}}
            west_go_left: {~SE2Transform: {theta_deg: 0}}
            north_go_left: {~SE2Transform: {theta_deg: -90}}
            north_go_right: {~SE2Transform: {theta_deg: -90}}
            east_go_right: {~SE2Transform: {theta_deg: 180}}
            east_go_straight: {~SE2Transform: {theta_deg: 180}}

3way_right:
    ~PlacedObject:  
        children:
            template: *3way_left           
        spatial_relations:
            template: {~SE2Transform: {theta_deg: 180}}   

# --------------------------------------------------------------------------------------------------------------------
# Double Lanes
# --------------------------------------------------------------------------------------------------------------------

go_right_double_right: &go_right_double_left
    ~LaneSegment:
      width: &width_double 0.188
      control_points: 
        - ~SE2Transform:
            p: [-0.50, -0.33]
            theta_deg: 0
        - ~SE2Transform:
            p: [-0.38, -0.38]
            theta_deg: -45
        - ~SE2Transform:
            p: [-0.33, -0.50]
            theta_deg: -90
            
go_right_double_left: &go_right_double_right
    ~LaneSegment:
      width: *width_double
      control_points: 
        - ~SE2Transform:
            p: [-0.50, -0.11]
            theta_deg: 0
        - ~SE2Transform:
            p: [-0.224, -0.224]
            theta_deg: -45
        - ~SE2Transform:
            p: [-0.11, -0.50]
            theta_deg: -90

go_right_double_right_to_left: &go_right_double_right_to_left
    ~LaneSegment:
      width: *width_double
      control_points: 
        - ~SE2Transform:
            p: [-0.50, -0.33]
            theta_deg: 0
        - ~SE2Transform:
            p: [-0.204, -0.33]
            theta_deg: 0
        - ~SE2Transform:
            p: [-0.138, -0.358]
            theta_deg: -45
        - ~SE2Transform:
            p: [-0.11, -0.424]
            theta_deg: -90
        - ~SE2Transform:
            p: [-0.11, -0.50]
            theta_deg: -90
            
            
go_right_double_left_to_right: &go_right_double_left_to_right
    ~LaneSegment:
      width: *width_double
      control_points: 
        - ~SE2Transform:
            p: [-0.50, -0.11]
            theta_deg: 0
        - ~SE2Transform:
            p: [-0.424, -0.11]
            theta_deg: 0
        - ~SE2Transform:
            p: [-0.358, -0.138]
            theta_deg: -45
        - ~SE2Transform:
            p: [-0.33, -0.204]
            theta_deg: -90
        - ~SE2Transform:
            p: [-0.33, -0.50]
            theta_deg: -90
            
go_straight_double_right: &go_straight_double_right
    ~LaneSegment:
      width: *width_double
      control_points: 
        - ~SE2Transform:
            p: [-0.5, -0.33]
            theta_deg: 0
        - ~SE2Transform:
            p: [+0.5, -0.33]
            theta_deg: 0
            
go_straight_double_left: &go_straight_double_left
    ~LaneSegment:
      width: *width_double
      control_points: 
        - ~SE2Transform:
            p: [-0.5, -0.11]
            theta_deg: 0
        - ~SE2Transform:
            p: [+0.5, -0.11]
            theta_deg: 0
            
go_straight_double_right_to_left: &go_straight_double_right_to_left
    ~LaneSegment:
      width: *width_double
      control_points: 
        - ~SE2Transform:
            p: [-0.50, -0.33]
            theta_deg: 0
        - ~SE2Transform:
            p: [-0.25, -0.33]
            theta_deg: 0
        - ~SE2Transform:
            p: [0, -0.22]
            theta_deg: 45
        - ~SE2Transform:
            p: [0.25, -0.11]
            theta_deg: 0
        - ~SE2Transform:
            p: [0.5, -0.11]
            theta_deg: 0
            
go_straight_double_left_to_right: &go_straight_double_left_to_right
    ~LaneSegment:
      width: *width_double
      control_points: 
        - ~SE2Transform:
            p: [-0.50, -0.11]
            theta_deg: 0
        - ~SE2Transform:
            p: [-0.25, -0.11]
            theta_deg: 0
        - ~SE2Transform:
            p: [0, -0.22]
            theta_deg: -45
        - ~SE2Transform:
            p: [0.25, -0.33]
            theta_deg: 0
        - ~SE2Transform:
            p: [0.5, -0.33]
            theta_deg: 0
            
go_left_double_right: &go_left_double_right
    ~LaneSegment:
      width: *width_double
      control_points: 
        - ~SE2Transform:
            p: [-0.50, -0.33]
            theta_deg: 0 

        - ~SE2Transform:
            p: [0.087, -0.087]
            theta_deg: 45

        - ~SE2Transform:
            p: [+0.33, +0.50]
            theta_deg: 90 
            
go_left_double_left: &go_left_double_left
    ~LaneSegment:
      width: *width_double
      control_points: 
        - ~SE2Transform:
            p: [-0.50, -0.11]
            theta_deg: 0 

        - ~SE2Transform:
            p: [-0.069, 0.069]
            theta_deg: 45

        - ~SE2Transform:
            p: [+0.11, +0.50]
            theta_deg: 90
            
go_left_double_right_to_left: &go_left_double_right_to_left
    ~LaneSegment:
      width: *width_double
      control_points: 
        - ~SE2Transform:
            p: [-0.50, -0.33]
            theta_deg: 0
        - ~SE2Transform:
            p: [0.016, -0.33]
            theta_deg: 0
        - ~SE2Transform:
            p: [0.082, -0.302]
            theta_deg: 45
        - ~SE2Transform:
            p: [0.11, -0.236]
            theta_deg: 90
        - ~SE2Transform:
            p: [0.11, 0.50]
            theta_deg: 90

go_left_double_left_to_right: &go_left_double_left_to_right
    ~LaneSegment:
      width: *width_double
      control_points: 
        - ~SE2Transform:
            p: [-0.50, -0.11]
            theta_deg: 0
        - ~SE2Transform:
            p: [0.236, -0.11]
            theta_deg: 0
        - ~SE2Transform:
            p: [0.302, -0.082]
            theta_deg: 45
        - ~SE2Transform:
            p: [0.33, -0.016]
            theta_deg: 90
        - ~SE2Transform:
            p: [0.33, 0.50]
            theta_deg: 90
            
straight_double:
    ~PlacedObject:
        children:
            lane1: *go_straight_double_right
            lane2: *go_straight_double_left
            lane3: *go_straight_double_right
            lane4: *go_straight_double_left
            lane5: *go_straight_double_right_to_left
            lane6: *go_straight_double_left_to_right
            lane7: *go_straight_double_right_to_left
            lane8: *go_straight_double_left_to_right
        spatial_relations:
            lane1: {~SE2Transform:}
            lane2: {~SE2Transform:}
            lane3: {~SE2Transform: {theta_deg: 180}}
            lane4: {~SE2Transform: {theta_deg: 180}}
            lane5: {~SE2Transform:}
            lane6: {~SE2Transform:}
            lane7: {~SE2Transform: {theta_deg: 180}}
            lane8: {~SE2Transform: {theta_deg: 180}}
            
curve_right_double: &curve_right_double
    ~PlacedObject:
        children:
            lane1_right: *go_right_double_right
            lane1_left: *go_right_double_left
            lane2_right: *go_left_double_right
            lane2_left: *go_left_double_left
        spatial_relations:
            lane1_right: {~SE2Transform:}
            lane1_left: {~SE2Transform:}
            lane2_right: {~SE2Transform: {theta_deg: 90}}
            lane2_left: {~SE2Transform: {theta_deg: 90}}


curve_left_double: &curve_left_double
    ~PlacedObject:
        children:
            curve: *curve_right_double
        spatial_relations: 
            curve: {~SE2Transform: {theta_deg: 270}}
            
1way_double: &1way_double
     ~PlacedObject:
        children:
            go_right_double_right: *go_right_double_right
            go_right_double_right_to_left: *go_right_double_right_to_left
            go_right_double_left: *go_right_double_left
            go_right_double_left_to_right: *go_right_double_left_to_right
            go_left_double_right: *go_left_double_right
            go_left_double_right_to_left: *go_left_double_right_to_left
            go_left_double_left: *go_left_double_left
            go_left_double_left_to_right: *go_left_double_left_to_right
            go_straight_double_right: *go_straight_double_right
            go_straight_double_right_to_left: *go_straight_double_right_to_left
            go_straight_double_left: *go_straight_double_left
            go_straight_double_left_to_right: *go_straight_double_left_to_right

4way_double:
    ~PlacedObject:
        children:
            a: *1way_double
            b: *1way_double
            c: *1way_double
            d: *1way_double
        spatial_relations:
            a: {~SE2Transform: {theta_deg: 0}}
            b: {~SE2Transform: {theta_deg: 90}}
            c: {~SE2Transform: {theta_deg: 180}}
            d: {~SE2Transform: {theta_deg: 270}}
            
3way_left_double: &3way_left_double
    ~PlacedObject:
        children:
            west_go_straight_double_right: *go_straight_double_right
            west_go_straight_double_left: *go_straight_double_left
            west_go_left_double_right: *go_left_double_right
            west_go_left_double_left: *go_left_double_left
            north_go_left_double_right: *go_left_double_right
            north_go_left_double_left: *go_left_double_left
            north_go_right_double_right: *go_right_double_right
            north_go_right_double_left: *go_right_double_left
            east_go_right_double_right: *go_right_double_right
            east_go_right_double_left: *go_right_double_left
            east_go_straight_double_right: *go_straight_double_right
            east_go_straight_double_left: *go_straight_double_left
        spatial_relations:
            west_go_straight_double_right: {~SE2Transform: {theta_deg: 0}}
            west_go_straight_double_left: {~SE2Transform: {theta_deg: 0}}
            west_go_left_double_right: {~SE2Transform: {theta_deg: 0}}
            west_go_left_double_left: {~SE2Transform: {theta_deg: 0}}
            north_go_left_double_right: {~SE2Transform: {theta_deg: -90}}
            north_go_left_double_left: {~SE2Transform: {theta_deg: -90}}
            north_go_right_double_right: {~SE2Transform: {theta_deg: -90}}
            north_go_right_double_left: {~SE2Transform: {theta_deg: -90}}
            east_go_right_double_right: {~SE2Transform: {theta_deg: 180}}
            east_go_right_double_left: {~SE2Transform: {theta_deg: 180}}
            east_go_straight_double_right: {~SE2Transform: {theta_deg: 180}}
            east_go_straight_double_left: {~SE2Transform: {theta_deg: 180}}

3way_right_double:
    ~PlacedObject:  
        children:
            template: *3way_left_double          
        spatial_relations:
            template: {~SE2Transform: {theta_deg: 180}} 
            
            
# -------------------------------------------------------------------------------------------------------------
# single lane tiles, only one direction (eg. for round-abouts)
# -------------------------------------------------------------------------------------------------------------

go_straight_single_right: &go_straight_single_right
    ~LaneSegment:
      width: *width
      control_points: 
        - ~SE2Transform:
            p: [-0.5, -0.204]
            theta_deg: 0
        - ~SE2Transform:
            p: [0.5, -0.204]
            theta_deg: 0
            
go_straight_single_left: &go_straight_single_left
    ~LaneSegment:
      width: *width
      control_points: 
        - ~SE2Transform:
            p: [-0.5, 0.204]
            theta_deg: 0
        - ~SE2Transform:
            p: [0.5, 0.204]
            theta_deg: 0
            
go_straight_single_right_to_left: &go_straight_single_right_to_left
    ~LaneSegment:
      width: *width
      control_points: 
        - ~SE2Transform:
            p: [-0.5, -0.204]
            theta_deg: 0
        - ~SE2Transform:
            p: [0, 0]
            theta_deg: 45
        - ~SE2Transform:
            p: [0.5, 0.204]
            theta_deg: 0
            
go_straight_single_left_to_right: &go_straight_single_left_to_right
    ~LaneSegment:
      width: *width
      control_points: 
        - ~SE2Transform:
            p: [-0.5, 0.204]
            theta_deg: 0
        - ~SE2Transform:
            p: [0, 0]
            theta_deg: -45
        - ~SE2Transform:
            p: [0.5, -0.204]
            theta_deg: 0
            
straight_single:
    ~PlacedObject:
        children:
            lane1: *go_straight_single_right
            lane2: *go_straight_single_left
            lane3: *go_straight_single_right_to_left
            lane4: *go_straight_single_left_to_right
        spatial_relations:
            lane1: {~SE2Transform:}
            lane2: {~SE2Transform:}
            lane3: {~SE2Transform:}
            lane4: {~SE2Transform:}

stay_in_roundabout_right: &stay_in_roundabout_right
    ~LaneSegment:
      width: *width
      control_points: 
        - ~SE2Transform:
            p: [-0.5, -0.304]
            theta_deg: 0
        - ~SE2Transform:
            p: [0.069, -0.069]
            theta_deg: 45
        - ~SE2Transform:
            p: [0.304, 0.5]
            theta_deg: 90

stay_in_roundabout_left: &stay_in_roundabout_left
    ~LaneSegment:
      width: *width
      control_points: 
        - ~SE2Transform:
            p: [-0.5, 0.104]
            theta_deg: 0
        - ~SE2Transform:
            p: [-0.22, 0.22]
            theta_deg: 45
        - ~SE2Transform:
            p: [-0.104, 0.5]
            theta_deg: 90
            
go_out_roundabout_left: &go_out_roundabout_left
    ~LaneSegment:
      width: *width
      control_points: 
        - ~SE2Transform:
            p: [-0.5, 0.104]
            theta_deg: 0
        - ~SE2Transform:
            p: [0, 0.154]
            theta_deg: 15
        - ~SE2Transform:
            p: [0.5, 0.204]
            theta_deg: 0
            
go_out_roundabout_right: &go_out_roundabout_right
    ~LaneSegment:
      width: *width
      control_points: 
        - ~SE2Transform:
            p: [-0.5, -0.304]
            theta_deg: 0
        - ~SE2Transform:
            p: [0, -0.254]
            theta_deg: 15
        - ~SE2Transform:
            p: [0.5, -0.204]
            theta_deg: 0
            
go_out_roundabout_left_to_right: &go_out_roundabout_left_to_right
    ~LaneSegment:
      width: *width
      control_points: 
        - ~SE2Transform:
            p: [-0.5, 0.104]
            theta_deg: 0
        - ~SE2Transform:
            p: [0, -0.05]
            theta_deg: -30
        - ~SE2Transform:
            p: [0.5, -0.204]
            theta_deg: 0
            
go_out_roundabout_right_to_left: &go_out_roundabout_right_to_left
    ~LaneSegment:
      width: *width
      control_points: 
        - ~SE2Transform:
            p: [-0.5, -0.304]
            theta_deg: 0
        - ~SE2Transform:
            p: [0, -0.05]
            theta_deg: 45
        - ~SE2Transform:
            p: [0.5, 0.204]
            theta_deg: 0
            
go_in_roundabout_left: &go_in_roundabout_left
    ~LaneSegment:
      width: *width
      control_points: 
        - ~SE2Transform:
            p: [-0.204, -0.5]
            theta_deg: 90
        - ~SE2Transform:
            p: [-0.154, 0]
            theta_deg: 75
        - ~SE2Transform:
            p: [-0.104, 0.5]
            theta_deg: 90
            
go_in_roundabout_right: &go_in_roundabout_right
    ~LaneSegment:
      width: *width
      control_points: 
        - ~SE2Transform:
            p: [0.204, -0.5]
            theta_deg: 90
        - ~SE2Transform:
            p: [0.254, 0]
            theta_deg: 75 
        - ~SE2Transform:
            p: [0.304, 0.5]
            theta_deg: 90

go_in_roundabout_left_to_right: &go_in_roundabout_left_to_right
    ~LaneSegment:
      width: *width
      control_points: 
        - ~SE2Transform:
            p: [-0.204, -0.5]
            theta_deg: 90
        - ~SE2Transform:
            p: [0.05, 0]
            theta_deg: 45 
        - ~SE2Transform:
            p: [0.304, 0.5]
            theta_deg: 90
            
go_in_roundabout_right_to_left: &go_in_roundabout_right_to_left
    ~LaneSegment:
      width: *width
      control_points: 
        - ~SE2Transform:
            p: [0.204, -0.5]
            theta_deg: 90
        - ~SE2Transform:
            p: [0.05, 0]
            theta_deg: 120
        - ~SE2Transform:
            p: [-0.104, 0.5]
            theta_deg: 90
            
roundabout_east: &roundabout_east
    ~PlacedObject:
        children:
            lane1: *stay_in_roundabout_right
            lane2: *stay_in_roundabout_left
            lane3: *go_out_roundabout_right
            lane4: *go_out_roundabout_left
            lane5: *go_out_roundabout_right_to_left
            lane6: *go_out_roundabout_left_to_right
            lane7: *go_in_roundabout_right
            lane8: *go_in_roundabout_left
            lane9: *go_in_roundabout_right_to_left
            lane10: *go_in_roundabout_left_to_right
        spatial_relations:
            lane1: {~SE2Transform:}
            lane2: {~SE2Transform:}
            lane3: {~SE2Transform:}
            lane4: {~SE2Transform:}
            lane5: {~SE2Transform:}
            lane6: {~SE2Transform:}
            lane7: {~SE2Transform:}
            lane8: {~SE2Transform:}
            lane9: {~SE2Transform:}
            lane10: {~SE2Transform:}
            
"""
"""
It contains all the tiles which can be used to build maps. Each tile contains lane segments
which are parametrized by control points (with orientation). This is an extension of the tiles available 
in duckietown_world.world_duckietown.tile_template.py
"""

@memoized_reset  # Wrapper to cache the results such that they will not be recomputed when run a second time
def load_driving_games_tile_types():
    """
    This functions converts the tiles from the yaml file to serializable python objects.
    This is a forked version of the function load_tile_types found in duckietown_world.world_duckietown.tile_template.py

    :return: Tiles as python objects (classes defined in the duckietown-world module)
    """
    s = yaml.load(data, Loader=yaml.SafeLoader)
    templates = Serializable.from_json_dict(s)
    return templates
