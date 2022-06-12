import itertools
from fractions import Fraction
from typing import Mapping

from frozendict import frozendict

from dg_commons import PlayerName
from dg_commons.planning import Trajectory
from driving_games.metrics_structures import PlayerOutcome
from possibilities import ProbDist
from preferences import FIRST_PREFERRED, SECOND_PREFERRED
from preferences.preferences_probability import ProbPreference


def get_joint_distribution_independent(distribution: Mapping[PlayerName, ProbDist]) -> ProbDist:
    def get_joint_values(distrib: Mapping[PlayerName, ProbDist]):
        keys, values = zip(*distrib.items())
        values_supp = (val.support() for val in values)
        permutations_dicts = [frozendict(zip(keys, v)) for v in itertools.product(*values_supp)]
        return frozenset(permutations_dicts)

    joint_values = get_joint_values(distribution)
    joint_distr = {}

    for joint_val in joint_values:
        prob_val = Fraction(1, 1)
        for player, val in joint_val.items():
            prob_val *= distribution[player].get(val)
        joint_distr[joint_val] = prob_val

    return ProbDist(joint_distr)


def never_second_preferred_stochastic(stochastic_preference: ProbPreference,
                                      outcome_distr: Mapping[Trajectory, ProbDist[PlayerOutcome]]):
    """
    Computes actions that are never second_preferred to any other action, by evaluating on stochastic outcomes.
    :param stochastic_preference:
    :param outcome_distr:
    :return: Set of actions that are not second preferred to any other action
    """

    all_player_actions = set(outcome_distr.keys())

    for traj_1, traj_2 in itertools.combinations(outcome_distr.keys(), 2):
        if traj_1 in all_player_actions and traj_2 in all_player_actions:
            result = stochastic_preference.compare(outcome_distr[traj_1], outcome_distr[traj_2])
            if result == FIRST_PREFERRED:
                all_player_actions.remove(traj_2)
            elif result == SECOND_PREFERRED:
                all_player_actions.remove(traj_1)

    return frozenset(all_player_actions)
