from frozendict import frozendict

from driving_games import UncertaintyParams
from games import GameSpec, Game, PlayerName, GamePlayer, get_accessible_states
from toy_games.toy_rewards import (
    BirdPersonalRewardStructureCustom,
    BirdPreferences,
    BirdJointReward,
)
from toy_games.toy_structures import FlyingDynamics, BirdState, BirdDirectObservations, BirdsVisualization
from possibilities import PossibilityMonad
from typing import FrozenSet as ASet, cast
from decimal import Decimal as D
from toy_games_tests.toy_games_tests_zoo import ToyGameMat

__all__ = ["get_toy_game_spec"]


def get_toy_game_spec(toy_game_mat: ToyGameMat, uncertainty_params: UncertaintyParams) -> GameSpec:
    ps: PossibilityMonad = uncertainty_params.poss_monad
    P1, P2 = PlayerName("1"), PlayerName("2")
    dt = D(1)  # not relevant for this example
    max_stages = toy_game_mat.get_max_stages()

    # state
    p1_x = BirdState()
    p2_x = BirdState()
    p1_initial = ps.unit(p1_x)
    p2_initial = ps.unit(p2_x)

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
    birds_joint_reward = BirdJointReward(
        max_stages=max_stages, subgames=toy_game_mat.subgames, row_player=P1, col_player=P2
    )

    p1 = GamePlayer(
        initial=p1_initial,
        dynamics=p1_dynamics,
        observations=p1_observations,
        personal_reward_structure=p1_personal_reward_structure,
        preferences=p1_preferences,
        monadic_preference_builder=mpref_builder,
    )
    p2 = GamePlayer(
        initial=p2_initial,
        dynamics=p2_dynamics,
        observations=p2_observations,
        personal_reward_structure=p2_personal_reward_structure,
        preferences=p2_preferences,
        monadic_preference_builder=mpref_builder,
    )

    handcrafted_game = Game(
        ps=ps,
        players=frozendict({P1: p1, P2: p2}),
        joint_reward=birds_joint_reward,
        game_visualization=BirdsVisualization(),
    )
    gs = GameSpec("Handcrafted game to study edge cases", handcrafted_game)
    return gs
