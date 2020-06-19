from frozendict import frozendict

from games import GameSpec, Game, PlayerName, GamePlayer, get_accessible_states
from toy_games.toy_rewards import (
    BirdPersonalRewardStructureCustom,
    BirdPreferences,
    BirdJointReward,
)
from toy_games.toy_structures import FlyingDynamics, BirdState, BirdDirectObservations
from possibilities import ProbabilitySet, PossibilityMonad
from typing import FrozenSet as ASet, cast, Sequence
from decimal import Decimal as D
import numpy as np
from preferences import SetPreference1

__all__ = ["get_toy_game_spec"]


def get_toy_game_spec(max_stages: int, leaves_payoffs: Sequence[np.ndarray]) -> GameSpec:
    ps: PossibilityMonad = ProbabilitySet()
    P1, P2 = PlayerName("1"), PlayerName("2")
    dt = D(1)  # not relevant for this example

    # state
    p1_x = BirdState()
    p2_x = BirdState()
    p1_initial = ps.unit(p1_x)
    p2_initial = ps.unit(p2_x)

    # dynamics
    p1_dynamics = FlyingDynamics()
    p2_dynamics = FlyingDynamics()

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
    set_preference_aggregator = SetPreference1
    birds_joint_reward = BirdJointReward(
        max_stages=max_stages, leaves_payoffs=leaves_payoffs, row_player=P1, col_player=P2
    )

    p1 = GamePlayer(
        initial=p1_initial,
        dynamics=p1_dynamics,
        observations=p1_observations,
        personal_reward_structure=p1_personal_reward_structure,
        preferences=p1_preferences,
        set_preference_aggregator=set_preference_aggregator,
    )
    p2 = GamePlayer(
        initial=p2_initial,
        dynamics=p2_dynamics,
        observations=p2_observations,
        personal_reward_structure=p2_personal_reward_structure,
        preferences=p2_preferences,
        set_preference_aggregator=set_preference_aggregator,
    )

    handcrafted_game = Game(
        ps=ps,
        players=frozendict({P1: p1, P2: p2}),
        joint_reward=birds_joint_reward,
        game_visualization=None
    )
    gs = GameSpec("Handcrafted game to study edge cases", handcrafted_game)
    return gs
