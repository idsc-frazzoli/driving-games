from frozendict import frozendict

from bayesian_driving_games.structures import BayesianGamePlayer, PlayerType
from driving_games import TwoVehicleUncertaintyParams, ProbPrefExpectedValue
from games import GameSpec, Game, PlayerName, GamePlayer, get_accessible_states
from nash import BiMatGame
from toy_games.bayesian_toy_joint_rewards import (
    BayesianBirdJointReward,
)
from toy_games.bayesian_toy_structures import BayesianFlyingDynamics
from toy_games.toy_rewards import (
    BirdPersonalRewardStructureCustom,
    BirdPreferences,
    BirdJointReward,
)
from toy_games.toy_structures import FlyingDynamics, BirdState, BirdDirectObservations, BirdsVisualization
from toy_games.bayesian_toy_structures import BayesianBirdState
from possibilities import PossibilitySet, PossibilityMonad, ProbabilityFraction
from typing import FrozenSet as ASet, cast, Sequence
from decimal import Decimal as D
import numpy as np
from preferences import SetPreference1

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
    p2_types = [PlayerType("neutral")]
    p1_types = [PlayerType("cautious"), PlayerType("aggressive")]

    # priors
    ps2 = ProbabilityFraction()
    p1_prior = ps2.lift_many(p2_types)
    p2_prior = ps2.lift_many(p1_types)

    # dynamics
    p1_dynamics = FlyingDynamics(poss_monad=ps)
    p2_dynamics = FlyingDynamics(poss_monad=ps)

    # personal reward structure
    p1_personal_reward_structure = BirdPersonalRewardStructureCustom(max_stages=max_stages)
    p2_personal_reward_structure = BirdPersonalRewardStructureCustom(max_stages=max_stages)

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
        types_of_other=p2_types,
        types_of_myself=p1_types,
        prior=p1_prior,
    )
    p2 = BayesianGamePlayer(
        initial=p2_initial,
        dynamics=p2_dynamics,
        observations=p2_observations,
        personal_reward_structure=p2_personal_reward_structure,
        preferences=p2_preferences,
        monadic_preference_builder=mpref_builder,
        types_of_other=p1_types,
        types_of_myself=p2_types,
        prior=p2_prior,
    )

    handcrafted_game = Game(
        ps=ps,
        players=frozendict({P1: p1, P2: p2}),
        joint_reward=birds_joint_reward,
        game_visualization=BirdsVisualization(),
    )
    gs = GameSpec("Handcrafted game to study edge cases", handcrafted_game)
    return gs
