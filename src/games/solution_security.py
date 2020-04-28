from typing import Dict, Mapping

from frozendict import frozendict

from .game_def import (
    JointMixedActions,
    JointPureActions,
    Outcome,
    PlayerName,
    SetOfOutcomes,
    U,
    ASet,
)
from .comb_utils import add_action, flatten_outcomes, get_all_combinations
from preferences import Preference, remove_dominated


def get_security_policies(
    solved: Mapping[JointPureActions, SetOfOutcomes],
    moves: JointMixedActions,
    preferences: Mapping[PlayerName, Preference[SetOfOutcomes]],
) -> JointMixedActions:
    actions: Dict[PlayerName, ASet[U]] = {}
    for player_name in moves:
        player_pref = preferences[player_name]
        actions[player_name] = get_security_policy(solved, moves, player_name, player_pref)

    return frozendict(actions)


def get_security_policy(
    solved: Mapping[JointPureActions, SetOfOutcomes],
    moves: JointMixedActions,
    player_name: PlayerName,
    preference: Preference[SetOfOutcomes],
) -> ASet[U]:
    others_choices = dict(moves)
    others_choices.pop(player_name)
    others_choices = frozendict(others_choices)
    preferences: Dict[PlayerName, Preference[SetOfOutcomes]]

    action2outcomes: Dict[U, ASet[Outcome]] = {}
    for player_action in moves[player_name]:
        option_outcomes = what_if_player_chooses(solved, player_name, player_action, others_choices)
        action2outcomes[player_action] = option_outcomes

    plausible = remove_dominated(action2outcomes, preference)
    return frozenset(plausible)
    # player_could_do: ASet[U]  =


def what_if_player_chooses(
    solved: Mapping[JointPureActions, SetOfOutcomes],
    player_name: PlayerName,
    player_action: U,
    others_choices: Mapping[PlayerName, ASet[U]],
) -> SetOfOutcomes:
    assert player_name not in others_choices
    """ Assume the player chooses u, and the others choose whatever. """
    set_pure_actions_others: ASet[JointPureActions]
    set_pure_actions_others = get_all_combinations(mixed_actions=others_choices)
    # now set also the player actions
    set_pure_actions_with_player = add_action(player_name, player_action, set_pure_actions_others)
    return flatten_outcomes(solved, set_pure_actions_with_player)
