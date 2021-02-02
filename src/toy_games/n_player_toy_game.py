from frozendict import frozendict
from dataclasses import dataclass
from typing import Dict, FrozenSet

from driving_games import UncertaintyParams
from games import Game, PlayerName, GamePlayer, get_accessible_states
from toy_games.n_player_toy_rewards import (
    ToyCarPersonalRewardStructureCustom,
    ToyCarPreferences,
    ToyCarJointReward,
    ToyCollision
)
from toy_games.n_player_toy_structures import (
    ToyCarDynamics,
    ToyCarState,
    ToyCarObservation,
    ToyCarDirectObservations,
    ToyCarVisualization,
    ToyCarActions,
    ToyCarCosts,
    ToyResources,
    ToyCarMap
)
from possibilities import PossibilityMonad
from typing import cast
from decimal import Decimal as D

__all__ = ["get_toy_car_game"]

ToyCarGame = Game[
    ToyCarState, ToyCarActions, ToyCarObservation, ToyCarCosts, ToyCollision, ToyResources
]

ToyCarGamePlayer = GamePlayer[
    ToyCarState, ToyCarActions, ToyCarObservation, ToyCarCosts, ToyCollision, ToyResources
]


@dataclass
class ToyGameParams:
    toy_game_map: ToyCarMap
    """ Map where the players play """
    max_wait: int
    """How long the players can maximally wait"""
    dt = D(1)
    """Not used"""


def get_toy_car_game(toy_games_params: ToyGameParams, uncertainty_params: UncertaintyParams) -> ToyCarGame:
    """
        Returns the game for n-player toy car game
    """

    ps: PossibilityMonad = uncertainty_params.poss_monad
    toy_map = toy_games_params.toy_game_map
    player_numbers = len(toy_map.lanes)
    player_names = [PlayerName(f"ToyCar_{i + 1}") for i in range(player_numbers)]

    toy_car_players: Dict[PlayerName, ToyCarGamePlayer] = {}

    for player, lane in zip(player_names, toy_map.lanes):

        toy_car_state = ToyCarState(
            along_lane=0,
            time=0,
            lane=lane,
            wait=0
        )

        toy_car_initial = ps.unit(toy_car_state)

        max_path = len(lane.control_points) - 1

        toy_car_dynamics = ToyCarDynamics(
            poss_monad=ps,
            max_path=max_path,
            max_wait=toy_games_params.max_wait
        )

        toy_car_personal_reward_structure = ToyCarPersonalRewardStructureCustom(
            max_path=max_path
        )
        toy_car_ac = get_accessible_states(toy_car_initial, toy_car_personal_reward_structure, toy_car_dynamics, D(1))

        toy_car_possible_states = cast(FrozenSet[ToyCarState], frozenset(toy_car_ac.nodes))

        # logger.info("npossiblestates", p1=len(p1_possible_states), p2=len(p2_possible_states))
        toy_car_observations = ToyCarDirectObservations(
            toy_car_possible_states, {player: toy_car_possible_states})

        toy_car_preferences = ToyCarPreferences()
        toy_car_player = GamePlayer(
            initial=toy_car_initial,
            dynamics=toy_car_dynamics,
            observations=toy_car_observations,
            personal_reward_structure=toy_car_personal_reward_structure,
            preferences=toy_car_preferences,
            monadic_preference_builder=uncertainty_params.mpref_builder,
        )

        toy_car_players[player] = toy_car_player

    joint_reward: ToyCarJointReward

    joint_reward = ToyCarJointReward()

    game_visualization: ToyCarVisualization

    game_visualization = ToyCarVisualization(
        toy_map=toy_map
    )
    game: ToyCarGame

    game = ToyCarGame(
        players=frozendict(toy_car_players),
        ps=ps,
        joint_reward=joint_reward,
        game_visualization=game_visualization,
    )
    return game
