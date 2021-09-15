from decimal import Decimal as D
from fractions import Fraction
from typing import cast, Dict, FrozenSet as ASet

from frozendict import frozendict

from _tmp.bayesian_driving_games.bayesian_driving_rewards import (
    BayesianVehicleJointReward,
    BayesianVehiclePersonalRewardStructureScalar,
)
from _tmp.bayesian_driving_games.structures import (
    BayesianVehicleState,
    BayesianGamePlayer,
    CAUTIOUS,
    AGGRESSIVE,
    NEUTRAL,
    BayesianGame,
)
from driving_games import TwoVehicleSimpleParams, VehicleTrackDynamics
from games import (
    GameVisualization,
    get_accessible_states,
    PlayerName,
    UncertaintyParams,
)
from possibilities import PossibilityMonad, PossibilityDist, ProbDist
from driving_games.collisions import Collision
from driving_games.preferences_coll_time import VehiclePreferencesCollTime
from driving_games.structures import (
    NO_LIGHTS,
    VehicleActions,
    VehicleCosts,
    VehicleGeometry,
    VehicleState,
)
from driving_games.vehicle_observation import VehicleDirectObservations, VehicleObservation
from driving_games.visualization import DrivingGameVisualization


# DrivingGame = Game[
#     BayesianVehicleState, VehicleActions, VehicleObservation, VehicleCosts, Collision, Rectangle
# ]
# DrivingGamePlayer = GamePlayer[
#     BayesianVehicleState, VehicleActions, VehicleObservation, VehicleCosts, Collision, Rectangle
# ]


def get_bayesian_driving_game(
    vehicles_params: TwoVehicleSimpleParams, uncertainty_params: UncertaintyParams
) -> BayesianGame:
    """

    :param vehicles_params: Vehicle parameters of the game
    :param uncertainty_params: uncertainty monad used
    :return: Returns a Bayesian DrivingGame, which includes types.
    """
    ps: PossibilityMonad = uncertainty_params.poss_monad
    L = vehicles_params.side + vehicles_params.road + vehicles_params.side
    start = vehicles_params.side + vehicles_params.road_lane_offset
    max_path = L - 1
    # p1_ref = SE2_from_xytheta([start, 0, np.pi / 2])
    p1_ref = (D(start), D(0), D(+90))
    # p2_ref = SE2_from_xytheta([L, start, -np.pi])
    p2_ref = (D(L), D(start), D(-180))
    max_speed = vehicles_params.max_speed
    min_speed = vehicles_params.min_speed
    max_wait = vehicles_params.max_wait
    dt = vehicles_params.dt
    available_accels = vehicles_params.available_accels

    P2 = PlayerName("W")
    P1 = PlayerName("N")
    mass = D(1000)
    length = D(4.5)
    width = D(1.8)

    # geometries
    g1 = VehicleGeometry(mass=mass, width=width, length=length, color=(1, 0, 0))
    g2 = VehicleGeometry(mass=mass, width=width, length=length, color=(0, 0, 1))
    geometries = {P1: g1, P2: g2}

    # types
    p1_types = {CAUTIOUS, AGGRESSIVE}
    p2_prior_weights = [Fraction(1, 2), Fraction(1, 2)]
    p2_types = {NEUTRAL}
    p1_prior_weights = [Fraction(1)]

    # priors
    ps2 = PossibilityDist()
    p1_prior_belief = {P2: ProbDist(dict(zip(p2_types, p1_prior_weights)))}
    p2_prior_belief = {P1: ProbDist(dict(zip(p1_types, p2_prior_weights)))}

    # State
    p1_x = VehicleState(
        ref=p1_ref,
        x=D(vehicles_params.first_progress),
        wait=D(0),
        v=D(0),
        light=NO_LIGHTS,
    )
    p1_initial = ps.unit(p1_x)
    p2_x = VehicleState(
        ref=p2_ref,
        x=D(vehicles_params.second_progress),
        wait=D(0),
        v=D(0),
        light=NO_LIGHTS,
    )
    p2_initial = ps.unit(p2_x)

    # Dynamics
    p1_dynamics = VehicleTrackDynamics(
        max_speed=max_speed,
        max_wait=max_wait,
        available_accels=available_accels,
        max_path=max_path,
        ref=p1_ref,
        lights_commands=vehicles_params.light_actions,
        min_speed=min_speed,
        vg=g1,
        shared_resources_ds=vehicles_params.shared_resources_ds,
        poss_monad=ps,
    )
    p2_dynamics = VehicleTrackDynamics(
        min_speed=min_speed,
        max_speed=max_speed,
        max_wait=max_wait,
        available_accels=available_accels,
        max_path=max_path,
        ref=p2_ref,
        lights_commands=vehicles_params.light_actions,
        vg=g2,
        shared_resources_ds=vehicles_params.shared_resources_ds,
        poss_monad=ps,
    )
    # todo check Personal reward
    p1_personal_reward_structure = BayesianVehiclePersonalRewardStructureScalar(max_path, p1_types)
    p2_personal_reward_structure = BayesianVehiclePersonalRewardStructureScalar(max_path, p2_types)

    g1 = get_accessible_states(p1_initial, p1_personal_reward_structure, p1_dynamics, dt)
    p1_possible_states = cast(ASet[BayesianVehicleState], frozenset(g1.nodes))
    # todo check why bayesian vehicle state
    g2 = get_accessible_states(p2_initial, p2_personal_reward_structure, p2_dynamics, dt)
    p2_possible_states = cast(ASet[BayesianVehicleState], frozenset(g2.nodes))

    # logger.info("npossiblestates", p1=len(p1_possible_states), p2=len(p2_possible_states))
    p1_observations = VehicleDirectObservations(p1_possible_states, {P2: p2_possible_states})
    p2_observations = VehicleDirectObservations(p2_possible_states, {P1: p1_possible_states})

    p1_preferences = VehiclePreferencesCollTime()  # fixme rename to CollScalar?
    p2_preferences = VehiclePreferencesCollTime()
    p1 = BayesianGamePlayer(
        initial=p1_initial,
        dynamics=p1_dynamics,
        observations=p1_observations,
        personal_reward_structure=p1_personal_reward_structure,
        preferences=p1_preferences,
        monadic_preference_builder=uncertainty_params.mpref_builder,
        types_of_myself=ps2.lift_many(p1_types),
        prior=p1_prior_belief,
    )
    p2 = BayesianGamePlayer(
        initial=p2_initial,
        dynamics=p2_dynamics,
        observations=p2_observations,
        personal_reward_structure=p2_personal_reward_structure,
        preferences=p2_preferences,
        monadic_preference_builder=uncertainty_params.mpref_builder,
        types_of_myself=ps2.lift_many(p2_types),
        prior=p2_prior_belief,
    )
    players: Dict[PlayerName, BayesianGamePlayer]
    players = {P1: p1, P2: p2}
    joint_reward: BayesianVehicleJointReward
    # todo check Joint reward
    joint_reward = BayesianVehicleJointReward(
        collision_threshold=vehicles_params.collision_threshold, geometries=geometries, players=players
    )

    game_visualization: GameVisualization[
        BayesianVehicleState, VehicleActions, VehicleObservation, VehicleCosts, Collision
    ]
    game_visualization = DrivingGameVisualization(
        vehicles_params, L, geometries=geometries, ds=vehicles_params.shared_resources_ds
    )

    game = BayesianGame(
        players=frozendict(players),
        ps=ps,
        joint_reward=joint_reward,
        game_visualization=game_visualization,
    )
    return game
