from frozendict import frozendict

from games import (
    GameSpec, Game,
    PlayerName, GamePlayer, get_accessible_states)
from handcrafted_games.handcrafted_structures import (
    FlyingDynamics, BirdState, BirdDirectObservations,
    BirdPersonalRewardStructureCustom, BirdPreferences)
from possibilities import PossibilityStructure, One, ProbabilitySet
from typing import FrozenSet as ASet, cast
from decimal import Decimal as D

from preferences import SetPreference1


def get_handcrafted_game_spec() -> GameSpec:
    ps: PossibilityStructure[One] = ProbabilitySet()
    P1, P2 = PlayerName("1"), PlayerName("2")
    dt = D(1)  # not relevant for this example
    max_stages: int = 2

    # state
    p1_x = BirdState()
    p2_x = BirdState()
    p1_initial = ps.lift_one(p1_x)
    p2_initial = ps.lift_one(p2_x)

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

    handcrafted_game = Game(ps=ps,
                            players=frozendict({P1: p1, P2: p2}),
                            joint_reward=,
                            game_visualization=None)
    gs = GameSpec("Handcrafted game to study edge cases",
                  handcrafted_game)
    return gs
