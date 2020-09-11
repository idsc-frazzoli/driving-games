from frozendict import frozendict

from driving_games import TwoVehicleUncertaintyParams
from games import GameSpec, Game, PlayerName, GamePlayer, get_accessible_states
from games.game_def import BayesianGamePlayer
from nash import BiMatGame
from toy_games.bayesian_toy_joint_rewards import BayesianBirdJointReward, BayesianBirdPersonalRewardStructureCustom
from toy_games.bayesian_toy_structures import BayesianFlyingDynamics
from toy_games.toy_rewards import (
    BirdPersonalRewardStructureCustom,
    BirdPreferences,
    BirdJointReward,
)
from toy_games.toy_structures import FlyingDynamics, BirdState, BirdDirectObservations, BirdsVisualization, \
    BayesianBirdState
from possibilities import PossibilitySet, PossibilityMonad
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
    p1_x = BayesianBirdState()
    p2_x = BayesianBirdState()
    p1_initial = ps.unit(p1_x)
    p2_initial = ps.unit(p2_x)

    # dynamics
    p1_dynamics = BayesianFlyingDynamics(poss_monad=ps)
    p2_dynamics = BayesianFlyingDynamics(poss_monad=ps)

    #types
    p1_types = {P1: ['cautious', 'aggressive']}
    p2_types = {P2: ['cautious', 'aggressive']}

    # personal reward structure
    p1_personal_reward_structure = BayesianBirdPersonalRewardStructureCustom(max_stages=max_stages)
    p2_personal_reward_structure = BayesianBirdPersonalRewardStructureCustom(max_stages=max_stages)

    # observations
    g1 = get_accessible_states(p1_initial, p1_personal_reward_structure, p1_dynamics, dt)
    p1_possible_states = cast(ASet[BayesianBirdState], frozenset(g1.nodes))
    g2 = get_accessible_states(p2_initial, p2_personal_reward_structure, p2_dynamics, dt)
    p2_possible_states = cast(ASet[BayesianBirdState], frozenset(g2.nodes))
    p1_observations = BirdDirectObservations(p1_possible_states, {P2: p2_possible_states})
    p2_observations = BirdDirectObservations(p2_possible_states, {P1: p1_possible_states})

    # preferences
    p1_preferences = BirdPreferences()
    p2_preferences = BirdPreferences()
    mpref_builder = uncertainty_params.mpref_builder
    birds_joint_reward = BayesianBirdJointReward(
        max_stages=max_stages, subgames=subgames, row_player=P1, col_player=P2, t1=p1_types, t2=p2_types
    )

    p1 = BayesianGamePlayer(
        initial=p1_initial,
        dynamics=p1_dynamics,
        observations=p1_observations,
        personal_reward_structure=p1_personal_reward_structure,
        preferences=p1_preferences,
        monadic_preference_builder=mpref_builder,
        types=p1_types
    )
    p2 = BayesianGamePlayer(
        initial=p2_initial,
        dynamics=p2_dynamics,
        observations=p2_observations,
        personal_reward_structure=p2_personal_reward_structure,
        preferences=p2_preferences,
        monadic_preference_builder=mpref_builder,
        types=p2_types
    )

    handcrafted_game = Game(
        ps=ps,
        players=frozendict({P1: p1, P2: p2}),
        joint_reward=birds_joint_reward,
        game_visualization=BirdsVisualization(),
    )
    gs = GameSpec("Handcrafted game to study edge cases", handcrafted_game)
    return gs
