from typing import Dict, List, Tuple
from math import pi
from typing import Set, Mapping
from decimal import Decimal as D

from games import PlayerName, MonadicPreferenceBuilder
from possibilities import PossibilitySet
from preferences import SetPreference1
from trajectory_games import (
    World,
    VehicleGeometry,
    VehicleState,
    SplinePathWithBounds,
    PathWithBounds,
    TrajectoryGamePlayer,
    TrajectoryGame,
    get_metrics_set,
    Metric,
    evaluate_metrics,
    PosetalPreference,
    TrajectoryGenerator1,
    TrajectoryParams,
)


def get_trajectory_game_players() -> Mapping[PlayerName, TrajectoryGamePlayer]:
    steps_dst, step_dst = 3, pi / 6
    steps_acc, step_acc = 3, 3.0
    u_acc = frozenset([D(_ * step_acc) for _ in range(-steps_acc // 2 + 1, steps_acc // 2 + 1)])
    u_dst = frozenset([D(_ * step_dst) for _ in range(-steps_dst // 2 + 1, steps_dst // 2 + 1)])

    param = TrajectoryParams(
        max_gen=1,
        dt=D("0.5"),
        u_acc=u_acc,
        u_dst=u_dst,
        v_max=D("20.0"),
        v_min=D("1.0"),
        st_max=D(pi / 4),
        vg=VehicleGeometry(m=D("200"), w=D("1"), l=D("1")),
    )
    traj_gen = TrajectoryGenerator1(params=param)

    p1 = PlayerName("Player1")
    p2 = PlayerName("Player2")
    state1 = VehicleState(x=D("0"), y=D("2.3"), th=D(pi / 2), v=D("8"), st=D("0"), t=D("0"))
    state2 = VehicleState(x=D("0"), y=D("0"), th=D(pi / 2), v=D("10"), st=D("0"), t=D("0"))

    ps = PossibilitySet()
    metrics: Set[Metric] = get_metrics_set()
    pref = PosetalPreference(keys=metrics)
    mpref_build: MonadicPreferenceBuilder = SetPreference1

    ret = {
        p1: TrajectoryGamePlayer(
            state=ps.unit(state1),
            actions_generator=traj_gen,
            preference=pref,
            monadic_preference_builder=mpref_build,
        ),
        p2: TrajectoryGamePlayer(
            state=ps.unit(state2),
            actions_generator=traj_gen,
            preference=pref,
            monadic_preference_builder=mpref_build,
        ),
    }
    return ret


def get_highway_world(players: Set[PlayerName]) -> World:
    s: List[D] = [D(_) for _ in range(20)]
    x: List[D] = [D(0.0) for _ in s]
    p_ref: List[Tuple[D, D]] = list(zip(x, s))
    p_left: List[Tuple[D, D]] = [(_, D(5.0)) for _ in s]
    p_right: List[Tuple[D, D]] = [(_, D(-5.0)) for _ in s]
    path = SplinePathWithBounds(s=s, p_ref=p_ref, p_left=p_left, p_right=p_right, bounds_sn=True)
    vg = VehicleGeometry(m=D("200"), w=D("1"), l=D("1"))
    ref: Dict[PlayerName, PathWithBounds] = {}
    geo: Dict[PlayerName, VehicleGeometry] = {}
    # TODO[SIR]: Extend to different paths for each player
    for player in players:
        ref[player] = path
        geo[player] = vg
    world = World(ref=ref, geo=geo)
    return world


def get_trajectory_game() -> TrajectoryGame:
    players: Mapping[PlayerName, TrajectoryGamePlayer] = get_trajectory_game_players()
    player_names: Set[PlayerName] = {_ for _ in players}
    world: World = get_highway_world(players=player_names)
    ps = PossibilitySet()

    game = TrajectoryGame(world=world, game_players=players, ps=ps, get_outcomes=evaluate_metrics)
    return game
