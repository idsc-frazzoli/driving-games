from typing import Dict, Mapping

from frozendict import frozendict

from dg_commons import RJ, RP, Y, U, X, PlayerName
from games.game_def import (
    UncertainCombined,
    SR,
    Game,
    MonadicPreferenceBuilder,
    Combined,
)
from possibilities import Poss
from preferences import Preference

__all__ = ["fd_r", "get_outcome_preferences_for_players", "add_incremental_cost_player"]


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


def add_incremental_cost_player(
    game: Game[X, U, Y, RP, RJ, SR],
    player_name: PlayerName,
    cur: Combined[RP, RJ],
    incremental: Poss[Mapping[PlayerName, Combined[RJ, RP]]],
) -> UncertainCombined:
    """Add incremental cost to a single player's outcome."""

    def add_incremental_cost(inc: Mapping[PlayerName, Combined[RJ, RP]]) -> Combined[RP, RJ]:
        pers_reduce = game.players[player_name].personal_reward_structure.personal_reward_reduce
        personal = pers_reduce(inc[player_name].personal, cur.personal)
        joint_reduce = game.joint_reward.joint_reward_reduce
        joint = joint_reduce(inc[player_name].joint, cur.joint)
        return Combined(personal=personal, joint=joint)

    return game.ps.build(incremental, add_incremental_cost)


def fd_r(d):
    """Freeze dictionary recursively. Assumes M[k1,M[k2, v]]"""
    return frozendict({k: frozendict(v) for k, v in d.items()})
