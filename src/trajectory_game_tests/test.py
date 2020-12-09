from typing import Dict, List, Tuple
from math import pi
from typing import Set
from decimal import Decimal as D

from games import PlayerName, MonadicPreferenceBuilder
from possibilities import PossibilitySet
from preferences import Preference, SetPreference1
from trajectory_game import TrajectoryParams, TrajectoryGenerator, Trajectory, World, VehicleGeometry, VehicleState, \
    MetricEvaluationResult, SplinePathWithBounds, PathWithBounds, \
    TrajectoryGamePlayer, TrajectoryGame, get_metrics_set, Metric, PlayerOutcome, compute_game_outcomes, \
    evaluate_metrics
from trajectory_game.preference import PosetalPreference


def get_trajectory_game_players() -> List[TrajectoryGamePlayer]:

    steps_dst, step_dst = 3, pi / 6
    steps_acc, step_acc = 3, 3.0
    u_acc = frozenset([D(_ * step_acc) for _ in range(-steps_acc // 2 + 1, steps_acc // 2 + 1)])
    u_dst = frozenset([D(_ * step_dst) for _ in range(-steps_dst // 2 + 1, steps_dst // 2 + 1)])

    param = TrajectoryParams(max_gen=1, dt=D('0.3'),
                             u_acc=u_acc, u_dst=u_dst,
                             v_max=D('20.0'), v_min=D('1.0'), st_max=D(pi / 4),
                             vg=VehicleGeometry(w=D('1'), lf=D('1'), lr=D('1')))
    gen = TrajectoryGenerator(params=param)

    p1 = PlayerName("Player1")
    p2 = PlayerName("Player2")
    state1 = VehicleState(x=D('0'), y=D('5'), th=D(pi / 2), v=D('8'), st=D('0'), t=D('0'))
    state2 = VehicleState(x=D('0'), y=D('0'), th=D(pi / 2), v=D('10'), st=D('0'), t=D('0'))

    ps = PossibilitySet()
    metrics: Set[Metric] = get_metrics_set()
    pref = PosetalPreference(keys=metrics)
    mpref_build: MonadicPreferenceBuilder = SetPreference1

    ret: List[TrajectoryGamePlayer] = [
        TrajectoryGamePlayer(name=p1, state=ps.unit(state1),
                             action_set_generator=gen,
                             preferences=pref,
                             monadic_preference_builder=mpref_build),
        TrajectoryGamePlayer(name=p2, state=ps.unit(state2),
                             action_set_generator=gen,
                             preferences=pref,
                             monadic_preference_builder=mpref_build),
    ]
    return ret


def create_highway_world(players: Set[PlayerName]) -> World:
    s: List[D] = [D(_) for _ in range(20)]
    x: List[D] = [D(0.0) for _ in s]
    p_ref: List[Tuple[D, D]] = list(zip(x, s))
    p_left: List[Tuple[D, D]] = [(_, D(5.0)) for _ in s]
    p_right: List[Tuple[D, D]] = [(_, D(-5.0)) for _ in s]
    path = SplinePathWithBounds(s=s, p_ref=p_ref, p_left=p_left,
                                p_right=p_right, bounds_sn=True)
    ref: Dict[PlayerName, PathWithBounds] = {}
    for player in players:
        ref[player] = path
    world = World(ref=ref)
    return world


def get_trajectory_game() -> TrajectoryGame:
    players: List[TrajectoryGamePlayer] = get_trajectory_game_players()
    player_names: Set[PlayerName] = {_.name for _ in players}
    world: World = create_highway_world(players=player_names)
    game_players: Dict[PlayerName, TrajectoryGamePlayer] = \
        {_.name: _ for _ in players}
    ps = PossibilitySet()

    game = TrajectoryGame(world=world, game_players=game_players, ps=ps,
                          game_outcomes=evaluate_metrics)
    return game


def test_trajectory_game():

    game: TrajectoryGame = get_trajectory_game()
    outcomes = compute_game_outcomes(traj_game=game)
    a = 2
