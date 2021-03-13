from decimal import Decimal as D
from dataclasses import replace

from games import PlayerName, UncertaintyParams
from possibilities import PossibilitySet, PossibilityDist
from preferences import SetPreference1
from preferences.preferences_probability import ProbPrefExpectedValue
from driving_games.structures import NO_LIGHTS

from duckie_games.game_generation import DuckieGameParams
from duckie_games.structures import DuckieGeometry

__all__ = [
    'two_player_4way',
    'two_player_4way_intersection_only',
    'three_player_4way',
    'three_player_4way_intersection_only',
    'three_player_4way_double',
    'three_player_4way_double_intersection_only',
    'uncertainty_sets',
    'uncertainty_prob'
]


uncertainty_sets = UncertaintyParams(poss_monad=PossibilitySet(), mpref_builder=SetPreference1)
uncertainty_prob = UncertaintyParams(poss_monad=PossibilityDist(), mpref_builder=ProbPrefExpectedValue)

#%%
"""
-----------------------------------------------------------------------
TWO PLAYER GAMES
-----------------------------------------------------------------------
"""
#%% Two player 4way

desc = (
    """
    Two player game on map 4way 
    """
)
map_name = '4way'
player_nb = 2
dt = D(1)
collision_threshold = 3

player_names = [PlayerName("Duckie_1"), PlayerName("Duckie_2")]

node_sequences = [
    ['P27', 'P4'],  # go straight
    ['P23', 'P26'],  # turn left
]

# node_sequences = [
#     ['P27', 'P4'],  # go straight
#     ['P23', 'P14']  # go straight
# ]

mass = D(1000)
length = D("4.5")
width = D("1.8")
height = D("1.8")
geometries = [
    DuckieGeometry(
        mass=mass,
        length=length,
        width=width,
        color=(1, 0, 0),
        height=height,
    ),
    DuckieGeometry(
            mass=mass,
            length=length,
            width=width,
            color=(0, 1, 0),
            height=height,
        )
]

shared_resources_ds = round(width / D(3), 2)

initial_progress = [D(1), D(1)]
max_paths = [D(20), D(20)]
max_speeds = [D(5), D(5)]
min_speeds = [D(1), D(1)]
max_waits = [D(1), D(1)]

# accel = [D(-1), D(0), D(+1)]
accel = [D(-2), D(-1), D(0), D(+1)]
# accel = [D(-2), D(-1), D(0), D(+0.5), D(+1)]
# accel = [D(-2), D(-1), D(-0.5), D(0), D(+0.5), D(+1)]
# accel = [D(-2), D(-1.5), D(-1), D(-0.5), D(0), D(+1.5), D(+1)]
# accel = [D(-2), D(-1.5), D(-1), D(-0.5), D(0), D(0.5), D(+1), D(+1.5)]
# accel = [D(-2), D(-1.5), D(-1), D(-0.5), D(-0.25), D(0), D(+0.5), D(1), D(+1.5)]
available_accels = [
    accel,
    accel
]
light_ac = [NO_LIGHTS]
light_actions = [
    light_ac,
    light_ac
]


player_geometries = {pn: _ for pn, _ in zip(player_names, geometries)}

player_max_paths = {pn: _ for pn, _ in zip(player_names, max_paths)}

player_max_speeds = {pn: _ for pn, _ in zip(player_names, max_speeds)}
player_min_speeds = {pn: _ for pn, _ in zip(player_names, min_speeds)}
player_max_waits = {pn: _ for pn, _ in zip(player_names, max_waits)}

player_node_sequence = {pn: _ for pn, _ in zip(player_names, node_sequences)}

# available_accels = {dn: frozenset([D(-2), D(-1), D(0), D(+1)]) for dn in player_names}
player_available_accels = {pn: frozenset(_) for pn, _ in zip(player_names, available_accels)}
player_light_actions = {pn: frozenset(_) for pn, _ in zip(player_names, light_actions)}

player_initial_progress = {pn: _ for pn, _ in zip(player_names, initial_progress)}

two_player_4way = DuckieGameParams(
    desc=desc,
    map_name=map_name,
    player_number=player_nb,
    player_names=player_names,
    player_geometries=player_geometries,
    max_speed=player_max_speeds,
    min_speed=player_min_speeds,
    max_wait=player_max_waits,
    max_path=player_max_paths,
    available_accels=player_available_accels,
    light_actions=player_light_actions,
    dt=dt,
    node_sequence=player_node_sequence,
    initial_progress=player_initial_progress,
    collision_threshold=collision_threshold,
    shared_resources_ds=shared_resources_ds
)

#%% 2 player 4 way intersection only
desc = (
    """
    Two player game on the map 4way intersection only
    """
)

node_sequences = [
    ['P15', 'P1'],  # go straight
    ['P13', 'P14'],  # turn left
]

# node_sequences = [
#     ['P15', 'P1'],  # go straight
#     ['P13', 'P5'],  # go straight
# ]

player_node_sequence = {pn: _ for pn, _ in zip(player_names, node_sequences)}
map_name = "4way-intersection-only"

two_player_4way_intersection_only = replace(
    obj=two_player_4way,
    desc=desc,
    map_name=map_name,
    node_sequence=player_node_sequence,
    lanes=None
)

#%% 2 player roundabout only
desc = (
    """
    Two player game on the map roundabout only
    """
)

node_sequences = [
    ['P18', 'P14', 'P10', 'P11'],
    ['P39', 'P33', 'P17', 'P14', 'P10'],
]

player_node_sequence = {pn: _ for pn, _ in zip(player_names, node_sequences)}

initial_progress = [D(0), D(0)]
max_paths = [D(20), D(20)]
max_speeds = [D(4), D(4)]
min_speeds = [D(1), D(1)]
max_waits = [D(1), D(1)]


# accel = [D(-1), D(0), D(+1)]
accel = [D(-1), D(0), D(+1), D(+1.5)]
# accel = [D(-2), D(-1), D(0), D(+0.5), D(+1)]
# accel = [D(-2), D(-1), D(-0.5), D(0), D(+0.5), D(+1)]
# accel = [D(-2), D(-1.5), D(-1), D(-0.5), D(0), D(+1.5), D(+1)]
# accel = [D(-2), D(-1.5), D(-1), D(-0.5), D(0), D(0.5), D(+1), D(+1.5)]
# accel = [D(-2), D(-1.5), D(-1), D(-0.5), D(-0.25), D(0), D(+0.5), D(1), D(+1.5)]
available_accels = [
    accel,
    accel
]
player_available_accels = {pn: frozenset(_) for pn, _ in zip(player_names, available_accels)}
player_initial_progress = {pn: _ for pn, _ in zip(player_names, initial_progress)}
player_max_paths = {pn: _ for pn, _ in zip(player_names, max_paths)}
player_max_speeds = {pn: _ for pn, _ in zip(player_names, max_speeds)}
player_min_speeds = {pn: _ for pn, _ in zip(player_names, min_speeds)}
player_max_waits = {pn: _ for pn, _ in zip(player_names, max_waits)}

map_name = "roundabout-only"

two_player_roundabout_only = replace(
    obj=two_player_4way,
    desc=desc,
    map_name=map_name,
    node_sequence=player_node_sequence,
    lanes=None,
    available_accels=player_available_accels,
    initial_progress=player_initial_progress,
    max_path=player_max_paths,
    max_speed=player_max_speeds,
    min_speed=player_min_speeds,
    max_wait=player_max_waits
)

#%%
"""
-----------------------------------------------------------------------
THREE PLAYER GAMES
-----------------------------------------------------------------------
"""
#%% 3 player 4way

desc = (
    """
    Three player game on the map 4way
    """
)

map_name = "4way"

player_nb = 3
dt = D(1)
collision_threshold = 3

player_names = [PlayerName("Duckie_1"), PlayerName("Duckie_2"), PlayerName("Duckie_3")]

node_sequences = [
    ['P27', 'P4'],  # go straight
    ['P23', 'P26'],  # turn left
    ['P13', 'P22']  # go straight
]

mass = D(1000)
length = D("4.5")
width = D("1.8")
height = D("1.8")
geometries = [
    DuckieGeometry(
        mass=mass,
        length=length,
        width=width,
        color=(1, 0, 0),
        height=height,
    ),
    DuckieGeometry(
            mass=mass,
            length=length,
            width=width,
            color=(0, 1, 0),
            height=height,
        ),
    DuckieGeometry(
            mass=mass,
            length=length,
            width=width,
            color=(0, 0, 1),
            height=height,
        )
]

shared_resources_ds = round(width / D(3), 2)

initial_progress = [D(1), D(1), D(1)]
max_paths = [D(20), D(20), D(20)]
max_speeds = [D(5), D(5), D(5)]
min_speeds = [D(1), D(1), D(1)]
max_waits = [D(1), D(1), D(1)]

# accel = [D(-1), D(0), D(+1)]
accel = [D(-2), D(-1), D(0), D(+1)]
# accel = [D(-2), D(-1), D(0), D(+0.5), D(+1)]
# accel = [D(-2), D(-1), D(-0.5), D(0), D(+0.5), D(+1)]
# accel = [D(-2), D(-1.5), D(-1), D(-0.5), D(0), D(+1.5), D(+1)]
# accel = [D(-2), D(-1.5), D(-1), D(-0.5), D(0), D(0.5), D(+1), D(+1.5)]
# accel = [D(-2), D(-1.5), D(-1), D(-0.5), D(-0.25), D(0), D(+0.5), D(1), D(+1.5)]
available_accels = [
    accel,
    accel,
    accel
]
light_ac = [NO_LIGHTS]
light_actions = [
    light_ac,
    light_ac,
    light_ac
]

player_geometries = {pn: _ for pn, _ in zip(player_names, geometries)}

player_max_paths = {pn: _ for pn, _ in zip(player_names, max_paths)}

player_max_speeds = {pn: _ for pn, _ in zip(player_names, max_speeds)}
player_min_speeds = {pn: _ for pn, _ in zip(player_names, min_speeds)}
player_max_waits = {pn: _ for pn, _ in zip(player_names, max_waits)}

player_node_sequence = {pn: _ for pn, _ in zip(player_names, node_sequences)}

# available_accels = {dn: frozenset([D(-2), D(-1), D(0), D(+1)]) for dn in player_names}
player_available_accels = {pn: frozenset(_) for pn, _ in zip(player_names, available_accels)}
player_light_actions = {pn: frozenset(_) for pn, _ in zip(player_names, light_actions)}

player_initial_progress = {pn: _ for pn, _ in zip(player_names, initial_progress)}

three_player_4way = DuckieGameParams(
    desc=desc,
    map_name=map_name,
    player_number=player_nb,
    player_names=player_names,
    player_geometries=player_geometries,
    max_speed=player_max_speeds,
    min_speed=player_min_speeds,
    max_wait=player_max_waits,
    max_path=player_max_paths,
    available_accels=player_available_accels,
    light_actions=player_light_actions,
    dt=dt,
    node_sequence=player_node_sequence,
    initial_progress=player_initial_progress,
    collision_threshold=collision_threshold,
    shared_resources_ds=shared_resources_ds
)

#%% 3 player 4way intersection only

desc = (
    """
    Three player game on the map 4way intersection only
    """
)

node_sequences = [
    ['P15', 'P1'],  # go straight
    ['P13', 'P14'],  # turn left
    ['P6', 'P12'],  # go straight
]


player_node_sequence = {pn: _ for pn, _ in zip(player_names, node_sequences)}
map_name = "4way-intersection-only"

three_player_4way_intersection_only = replace(
    obj=three_player_4way,
    desc=desc,
    map_name=map_name,
    node_sequence=player_node_sequence,
    lanes=None
)

#%% 3 player 4 way double

desc = (
    """
    Three player game on the map 4way double
    """
)

map_name = "4way-double"

player_nb = 3
dt = D(1)
collision_threshold = 3

player_names = [PlayerName("Duckie_1"), PlayerName("Duckie_2"), PlayerName("Duckie_3")]

node_sequences = [
    ['P74', 'P56', 'P48', 'P44'],  # turn left
    ['P63', 'P58', 'P49', 'P45'],  # go straight
    ['P42', 'P50', 'P32', 'P16']  # turn left
]

mass = D(1000)
length = D("4.5")
width = D("1.8")
height = D("1.8")
geometries = [
    DuckieGeometry(
        mass=mass,
        length=length,
        width=width,
        color=(1, 0, 0),
        height=height,
    ),
    DuckieGeometry(
            mass=mass,
            length=length,
            width=width,
            color=(0, 1, 0),
            height=height,
        ),
    DuckieGeometry(
            mass=mass,
            length=length,
            width=width,
            color=(0, 0, 1),
            height=height,
        )
]

shared_resources_ds = round(width / D(3), 2)

initial_progress = [D(10), D(10), D(10)]
max_paths = [D(30), D(30), D(30)]
max_speeds = [D(5), D(5), D(5)]
min_speeds = [D(1), D(1), D(1)]
max_waits = [D(1), D(1), D(1)]

accel = [D(-2), D(-1), D(0), D(+1)]
# accel = [D(-1), D(0), D(+1)]
# accel = [D(0), D(+3)]
available_accels = [
    accel,
    accel,
    accel
]
light_ac = [NO_LIGHTS]
light_actions = [
    light_ac,
    light_ac,
    light_ac
]


player_geometries = {pn: _ for pn, _ in zip(player_names, geometries)}

player_max_paths = {pn: _ for pn, _ in zip(player_names, max_paths)}

player_max_speeds = {pn: _ for pn, _ in zip(player_names, max_speeds)}
player_min_speeds = {pn: _ for pn, _ in zip(player_names, min_speeds)}
player_max_waits = {pn: _ for pn, _ in zip(player_names, max_waits)}

player_node_sequence = {pn: _ for pn, _ in zip(player_names, node_sequences)}

# available_accels = {dn: frozenset([D(-2), D(-1), D(0), D(+1)]) for dn in player_names}
player_available_accels = {pn: frozenset(_) for pn, _ in zip(player_names, available_accels)}
player_light_actions = {pn: frozenset(_) for pn, _ in zip(player_names, light_actions)}

player_initial_progress = {pn: _ for pn, _ in zip(player_names, initial_progress)}

three_player_4way_double = DuckieGameParams(
    desc=desc,
    map_name=map_name,
    player_number=player_nb,
    player_names=player_names,
    player_geometries=player_geometries,
    max_speed=player_max_speeds,
    min_speed=player_min_speeds,
    max_wait=player_max_waits,
    max_path=player_max_paths,
    available_accels=player_available_accels,
    light_actions=player_light_actions,
    dt=dt,
    node_sequence=player_node_sequence,
    initial_progress=player_initial_progress,
    collision_threshold=collision_threshold,
    shared_resources_ds=shared_resources_ds
)

#%% 3 player 4way double intersection only

desc = (
    """
    Three player game on the map 4way double intersection only
    """
)

node_sequences = [
    ['P30', 'P20', 'P8', 'P9'],  # turn left
    ['P27', 'P22', 'P10', 'P11'],  # go straight
    ['P12', 'P13', 'P0', 'P1']  # turn left
]


player_node_sequence = {pn: _ for pn, _ in zip(player_names, node_sequences)}
map_name = "4way-double-intersection-only"

three_player_4way_double_intersection_only = replace(
    obj=three_player_4way_double,
    desc=desc,
    map_name=map_name,
    node_sequence=player_node_sequence,
    lanes=None
)
