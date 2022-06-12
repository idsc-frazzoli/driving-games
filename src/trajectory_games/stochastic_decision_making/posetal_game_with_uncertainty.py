import copy
from typing import Mapping, TypeVar

from frozendict import frozendict

from dg_commons import PlayerName
from dg_commons.planning import JointTrajectories
from driving_games.metrics_structures import JointPlayerOutcome
from possibilities import ProbDist
from preferences import Preference, SECOND_PREFERRED, FIRST_PREFERRED
from trajectory_games.stochastic_decision_making.utils import get_joint_distribution_independent

P = TypeVar("P")


class StochasticNEComputation:
    """
    Class to compute NE where actions have uncertain outcomes.
    """

    def __init__(self,
                 pref_distr: Mapping[PlayerName, ProbDist[Preference]],
                 joint_actions_outcomes_mapping: Mapping[JointTrajectories, JointPlayerOutcome],
                 ):

        self.pref_distr = pref_distr
        self.joint_actions_outcomes_mapping = joint_actions_outcomes_mapping


    def _get_all_actions(self):
        """
        :return: Dictionary containing available actions for each player
        """
        actions = {}
        for player in self.pref_distr.keys():
            actions[player] = copy.deepcopy(set())

        for player in actions.keys():
            for joint_action in self.joint_actions_outcomes_mapping.keys():
                actions[player].add(joint_action[player])

        return actions

    def brute_force_NE(self, preferences: Mapping[PlayerName, Preference]):
        """
        This function will compute brute force NE, i.e. checking if a joint action is a NE by comparing to all
        alternatives.
        GIVEN: Outcome for each action, as a distribution
        :param method: method for to compare distributions of partial orders
        :return:
        """

        strong_eqs = set()
        weak_eqs = set()

        all_actions = self._get_all_actions()

        # loop over joint trajectories
        for joint_traj in self.joint_actions_outcomes_mapping.keys():
            comparisons = set()

            # unfreeze dict
            alt_joint_traj = {}
            for key, value in joint_traj.items():
                alt_joint_traj[key] = value

            # for each player, check if current action is best response
            for player in joint_traj.keys():
                outcome = self.joint_actions_outcomes_mapping[joint_traj][player]
                # compare with all other available actions
                for action in all_actions[player]:
                    if action == joint_traj[player]:
                        continue
                    alt_joint_traj[player] = action
                    alternative_outcome = self.joint_actions_outcomes_mapping[frozendict(alt_joint_traj)][player]
                    comparison_outcome = preferences[player].compare(alternative_outcome, outcome)
                    comparisons.add(comparison_outcome)
                alt_joint_traj[player] = joint_traj[player]
                # todo: check if correct

            # current joint trajectory is strong NE
            if len(comparisons) == 1 and SECOND_PREFERRED in comparisons:
                strong_eqs.add(joint_traj)

            # current joint trajectory is weak NE
            if FIRST_PREFERRED not in comparisons:
                weak_eqs.add(joint_traj)

        return strong_eqs, weak_eqs

    def NE_distr(self):
        """
        Find distribution of NE for a player holding beliefs on preference of other players.
        :return:
        """

        NE_prob_dict = {}
        # assume preferences are independent
        joint_pref_distr = get_joint_distribution_independent(self.pref_distr)
        for joint_pref in joint_pref_distr.support():
            strong_eqs, weak_eqs = self.brute_force_NE(joint_pref)
            NE = {"weak_eqs": frozenset(weak_eqs), "strong_eqs": frozenset(strong_eqs)}
            NE_prob_dict[frozendict(NE)] = joint_pref_distr.get(joint_pref)

        return ProbDist(NE_prob_dict)


