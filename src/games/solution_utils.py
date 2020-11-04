from typing import Dict, Mapping

from frozendict import frozendict

from games.game_def import (
    PlayerName,
    UncertainCombined,
    SR,
    RJ,
    RP,
    Y,
    U,
    X,
    Game,
    MonadicPreferenceBuilder,
    Combined,
)
from possibilities import Poss
from preferences import Preference

__all__ = ["fr", "get_outcome_preferences_for_players", "add_incremental_cost_single"]


def get_outcome_preferences_for_players(
    game: Game[X, U, Y, RP, RJ, SR],
) -> Mapping[PlayerName, Preference[UncertainCombined]]:
    """

    :param game:
    :return:
    """
    preferences: Dict[PlayerName, Preference[UncertainCombined]] = {}
    for player_name, player in game.players.items():
        pref0: Preference[Combined[RJ, RP]] = player.preferences
        monadic_pref_builder: MonadicPreferenceBuilder
        monadic_pref_builder = player.monadic_preference_builder
        pref2: Preference[UncertainCombined] = monadic_pref_builder(pref0)
        preferences[player_name] = pref2
    return preferences


def add_incremental_cost_single(
    game: Game[X, U, Y, RP, RJ, SR],
    *,
    player_name: PlayerName,
    cur: Combined[RP, RJ],
    incremental_for_player: Mapping[PlayerName, Poss[RP]],
) -> Combined[RP, RJ]:
    inc = incremental_for_player[player_name]
    reduce = game.players[player_name].personal_reward_structure.personal_reward_reduce
    personal = reduce(inc, cur.personal)

    joint = cur.joint
    return Combined(personal=personal, joint=joint)


def fr(d):
    return frozendict({k: frozendict(v) for k, v in d.items()})
