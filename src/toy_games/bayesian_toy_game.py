from fractions import Fraction

from frozendict import frozendict

from bayesian_driving_games.structures import BayesianGamePlayer, NEUTRAL, CAUTIOUS, AGGRESSIVE, BayesianGame
from driving_games import TwoVehicleUncertaintyParams
from games import GameSpec, Game, PlayerName, get_accessible_states
from nash import BiMatGame
from toy_games.bayesian_toy_rewards import (
    BayesianBirdJointReward,
    BayesianBirdPersonalReward,
)
from toy_games.toy_rewards import (
    BirdPreferences,
)
from toy_games.toy_structures import FlyingDynamics, BirdState, BirdDirectObservations, BirdsVisualization
from possibilities import PossibilityMonad, PossibilityDist, ProbDist
from typing import FrozenSet as ASet, cast, Sequence
from decimal import Decimal as D

__all__ = ["get_bayesian_toy_game_spec"]


def get_bayesian_toy_game_spec(
    max_stages: int, subgames: Sequence[BiMatGame], uncertainty_params: TwoVehicleUncertaintyParams
) -> GameSpec:
    ps: PossibilityMonad = uncertainty_params.poss_monad
    P1, P2 = PlayerName("1"), PlayerName("2")
    dt = D(1)  # not relevant for this example

    # state
    p1_x = BirdState()
    p2_x = BirdState()
    p1_initial = ps.unit(p1_x)
    p2_initial = ps.unit(p2_x)

    # types
    p1_types = {CAUTIOUS, AGGRESSIVE}
    p2_prior_weights = [Fraction(1, 2), Fraction(1, 2)]
    p2_types = {NEUTRAL}
    p1_prior_weights = [Fraction(1)]

    # priors
    ps2 = PossibilityDist()
    p1_prior_belief = {P2: ProbDist(dict(zip(p2_types, p1_prior_weights)))}
    p2_prior_belief = {P1: ProbDist(dict(zip(p1_types, p2_prior_weights)))}

    # dynamics
    p1_dynamics = FlyingDynamics(poss_monad=ps)
    p2_dynamics = FlyingDynamics(poss_monad=ps)

    # personal reward structure
    p1_personal_reward_structure = BayesianBirdPersonalReward(max_stages=max_stages, p_types=p1_types)
    p2_personal_reward_structure = BayesianBirdPersonalReward(max_stages=max_stages, p_types=p2_types)

    # observations
    g1 = get_accessible_states(p1_initial, p1_personal_reward_structure, p1_dynamics, dt)
    p1_possible_states = cast(ASet[BirdState], frozenset(g1.nodes))
    g2 = get_accessible_states(p2_initial, p2_personal_reward_structure, p2_dynamics, dt)
    p2_possible_states = cast(ASet[BirdState], frozenset(g2.nodes))
    p1_observations = BirdDirectObservations(p1_possible_states, {P2: p2_possible_states})
    p2_observations = BirdDirectObservations(p2_possible_states, {P1: p1_possible_states})

    # preferences
    p1_preferences = BirdPreferences()
    p2_preferences = BirdPreferences()
    mpref_builder = uncertainty_params.mpref_builder
    birds_joint_reward = BayesianBirdJointReward(
        max_stages=max_stages,
        subgames=subgames,
        row_player=P1,
        col_player=P2,
        p1_types=p1_types,
        p2_types=p2_types,
    )

    p1 = BayesianGamePlayer(
        initial=p1_initial,
        dynamics=p1_dynamics,
        observations=p1_observations,
        personal_reward_structure=p1_personal_reward_structure,
        preferences=p1_preferences,
        monadic_preference_builder=mpref_builder,
        types_of_myself=ps2.lift_many(p1_types),
        prior=p1_prior_belief,
    )
    p2 = BayesianGamePlayer(
        initial=p2_initial,
        dynamics=p2_dynamics,
        observations=p2_observations,
        personal_reward_structure=p2_personal_reward_structure,
        preferences=p2_preferences,
        monadic_preference_builder=mpref_builder,
        types_of_myself=ps2.lift_many(p2_types),
        prior=p2_prior_belief,
    )

    handcrafted_game = BayesianGame(
        ps=ps,
        players=frozendict({P1: p1, P2: p2}),
        joint_reward=birds_joint_reward,
        game_visualization=BirdsVisualization(),
    )
    return GameSpec("Bayesian handcrafted game", handcrafted_game)
