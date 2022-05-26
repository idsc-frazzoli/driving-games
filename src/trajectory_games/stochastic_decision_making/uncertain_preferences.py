import copy
from collections import defaultdict
from fractions import Fraction
from typing import Mapping

import frozendict
from frozendict import frozendict

from dg_commons import PlayerName
from dg_commons.planning import JointTrajectories
from driving_games.metrics_structures import JointPlayerOutcome
from possibilities import ProbDist
from preferences import Preference
from preferences.preferences_probability import ProbPrefExpectedValueMetricsDict, ProbPrefMostLikely
from .outcome_distributions import OutcomeDistributionGenerator
from .utils import get_joint_distribution_independent, never_second_preferred_stochastic


class UncertainPreferenceOutcomes:
    """
    Class to compute outcome distributions while holding a belief over preferences of other agents.
    Assuming beliefs over preferences are independent.
    """

    def __init__(self,
                 my_name: PlayerName,
                 pref_distr: Mapping[PlayerName, ProbDist[Preference]],
                 joint_actions_outcomes_mapping: Mapping[JointTrajectories, JointPlayerOutcome],
                 ego_pref: Preference
                 ):

        self.my_name = my_name  # also referred to as "ego" in the code
        self.joint_pref_distr = get_joint_distribution_independent(pref_distr)
        self.joint_actions_outcomes_mapping = joint_actions_outcomes_mapping
        self.my_pref = ego_pref

    def outcome_distr(self):
        """
        Compute outcome distributions for each action available to player "my_name", by considering all elements
        of the support of a joint distribution over preferences of other players
        :return: Distribution over outcomes for each action available to player "my_name"
        """

        helper_dict = {}
        # compute outcome distribution for each action and for each preference combination
        for prefs, prob in self.joint_pref_distr.it():
            # unfreeze dict to add ego preference
            new_pref = {}
            for key, value in prefs.items():
                new_pref[key] = value
            new_pref[self.my_name] = self.my_pref

            outcome_distr_gen = OutcomeDistributionGenerator(self.my_name,
                                                             frozendict(new_pref),
                                                             self.joint_actions_outcomes_mapping)

            outcome_distr = outcome_distr_gen.get_ego_outcome_distributions()

            for action, distr in outcome_distr.items():
                if action not in helper_dict.keys():
                    helper_dict[action] = []
                helper_dict[action].append((prob, distr))

        # combine results of previous step
        distr_of_distr = {}
        for action, tuples_list in helper_dict.items():
            total_prob = 0.0
            # for fraction, _ in tuples_list:

            action_p = {}
            for fraction, prob_distr in tuples_list:
                total_prob += fraction
                action_p[frozendict(prob_distr.p)] = 0.0
            for fraction, prob_distr in tuples_list:
                action_p[frozendict(prob_distr.p)] += (fraction / total_prob)
            distr_of_distr[action] = ProbDist(p=copy.deepcopy(action_p))

        final_distr_of_distr = {}

        # compute distribution of distributions (the join)
        for action, distr in distr_of_distr.items():
            res = defaultdict(Fraction)
            for distr_dict, weight in distr.it():
                for key, value in distr_dict.items():
                    res[key] += value * weight
            final_distr_of_distr[action] = ProbDist(res)

        return final_distr_of_distr

    def action_selector(self, method: str):
        """
        Select an action from a set of actions where for each action, a distribution over outcomes is given.

        :param method:  method for action selection from distributions of outcomes
                        avg: compare the average outcome for each distribution
                        argmax: compares the most likely outcome for each distribution
                        stochastic_dominance: compare distributions by stochastic dominance:
                                              https://en.wikipedia.org/wiki/Stochastic_dominance

        :return:        best action(s) according to method. Best:
        """
        outcome_distr_ego = self.outcome_distr()

        if method == "avg" or method == "average":
            ego_prob_pref_avg = ProbPrefExpectedValueMetricsDict(p0=self.my_pref)
            not_second_pref_actions = never_second_preferred_stochastic(ego_prob_pref_avg, outcome_distr_ego)

        elif method == "argmax":
            ego_prob_pref_argmax = ProbPrefMostLikely(p0=self.my_pref)
            not_second_pref_actions = never_second_preferred_stochastic(ego_prob_pref_argmax, outcome_distr_ego)

        elif method == "stochastic_dominance":
            raise NotImplementedError
        else:
            raise NotImplementedError

        return not_second_pref_actions
