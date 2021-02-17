from typing import Dict, List
from math import pi
from typing import Set, Mapping
from decimal import Decimal as D

from games import PlayerName, MonadicPreferenceBuilder
from possibilities import PossibilitySet
from preferences import SetPreference1
from trajectory_games import (
    TrajectoryWorld,
    VehicleGeometry,
    VehicleState,
    TrajectoryGamePlayer,
    TrajectoryGame,
    get_metrics_set,
    Metric,
    evaluate_metrics,
    PosetalPreference,
    TrajectoryGenerator1,
    TrajectoryParams,
    TrajGameVisualization,
)
from world import NodeName, SE2Transform

__all__ = [
    "get_trajectory_game",
]


def get_trajectory_game_players(world: TrajectoryWorld) ->\
        Mapping[PlayerName, TrajectoryGamePlayer]:
    steps_dst, step_dst = 3, pi / 30.0
    steps_acc, step_acc = 1, 3.0
    u_acc = frozenset([D(_ * step_acc) for _ in range(-steps_acc // 2 + 1, steps_acc // 2 + 1)])
    u_dst = frozenset([D(_ * step_dst) for _ in range(-steps_dst // 2 + 1, steps_dst // 2 + 1)])
    v0  = D("10")    # Initial velocity
    st0 = D("0")    # Initial steering
    t0  = D("0")    # Initial time
    s0 = 6.0        # Initial progress

    vg: VehicleGeometry = world.get_geometry(world.get_players()[0])
    param = TrajectoryParams(
        max_gen=2,
        dt=D("1.0"),
        u_acc=u_acc,
        u_dst=u_dst,
        v_max=D("20.0"),
        v_min=D("1.0"),
        st_max=D(pi / 4),
        vg=vg,
    )
    traj_gen = TrajectoryGenerator1(params=param)

    ps = PossibilitySet()
    metrics: Set[Metric] = get_metrics_set()
    mpref_build: MonadicPreferenceBuilder = SetPreference1
    ret: Dict[PlayerName, TrajectoryGamePlayer] = {}
    for player in world.get_players():
        lane = world.get_lane(player)
        beta0 = lane.beta_from_along_lane(along_lane=s0)
        se2 = SE2Transform.from_SE2(lane.center_point(beta=beta0))
        state = VehicleState(x=D(se2.p[0]), y=D(se2.p[1]), th=D(se2.theta), v=v0, st=st0, t=t0)
        pref = PosetalPreference(pref_file=f"player_pref/{player}.pref", keys=metrics)
        ret[player] = TrajectoryGamePlayer(state=ps.unit(state), actions_generator=traj_gen,
                                           preference=pref, monadic_preference_builder=mpref_build,
                                           vg=world.get_geometry(player))
    return ret


def get_4way_double_world(names: List[PlayerName]) -> TrajectoryWorld:
    assert len(names) <= 4, "Max 4 player sequences defined for game"
    map_name = '4way-double-intersection-only'
    node_sequences = [
        ['P30', 'P21', 'P8',  'P9' ],   # down, left
        ['P27', 'P22', 'P10', 'P11'],   # right, left
        ['P12', 'P13', 'P0',  'P1' ],   # left, up
        ['P4',  'P5',  'P17', 'P28'],   # up, down
    ]
    colours = [
        (1, 0, 0),
        (0, 0, 1),
        (0, 0.6, 0),
        (1, 0.7, 0),
    ]
    geometries = [VehicleGeometry(m=D("200"), w=D("0.7"), l=D("1.5"), colour=col) for col in colours]
    ref: Dict[PlayerName, List[NodeName]] = {}
    geo: Dict[PlayerName, VehicleGeometry] = {}
    i = 0
    for p_name in names:
        ref[p_name] = node_sequences[i]
        geo[p_name] = geometries[i]
        i = i+1
    world = TrajectoryWorld(map_name=map_name, geo=geo, nodes=ref)
    return world


def get_trajectory_game() -> TrajectoryGame:
    player_names: List[PlayerName] = [
        PlayerName("P1"),
        PlayerName("P2"),
        PlayerName("P3"),
        PlayerName("P4"),
    ]
    world: TrajectoryWorld = get_4way_double_world(names=player_names)
    players: Mapping[PlayerName, TrajectoryGamePlayer] = \
        get_trajectory_game_players(world=world)
    ps = PossibilitySet()

    game = TrajectoryGame(world=world, game_players=players, ps=ps,
                          get_outcomes=evaluate_metrics,
                          game_vis=TrajGameVisualization(world=world))
    return game
