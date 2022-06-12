import copy
from copy import deepcopy
from fractions import Fraction
from itertools import combinations
from typing import Mapping, TypeVar

from dg_commons import PlayerName
from dg_commons.planning import Trajectory, JointTrajectories
from driving_games.metrics_structures import JointPlayerOutcome, PlayerOutcome, EvaluatedMetric, MetricNodeName
from possibilities import ProbDist
from possibilities.prob import variable_change
from preferences import Preference, SECOND_PREFERRED, FIRST_PREFERRED
from preferences.preferences_probability import ProbPrefExpectedValueMetricsDict, ProbPrefMostLikely
from .utils import get_joint_distribution_independent, never_second_preferred_stochastic

P = TypeVar("P")


class OutcomeDistributionGenerator:
    """
    Class to generate outcome distributions from preferences and available actions.
    """

    def __init__(
            self,
            my_name: PlayerName,
            preferences: Mapping[PlayerName, Preference[P]],
            joint_actions_outcomes_mapping: Mapping[JointTrajectories, JointPlayerOutcome]

    ):
        self.my_name = my_name  # also referred to as "ego" in the code
        self.preferences = preferences
        self.joint_actions_outcomes_mapping = joint_actions_outcomes_mapping
        self.ego_actions = set()

        self.aggregated_outcomes: Mapping[PlayerName, Mapping[Trajectory, EvaluatedMetric]] = {}
        # compute aggregated outcomes for all players/{Ego}
        self._aggregate_all_outcomes()
        assert self.my_name in self.preferences.keys(), "my_name is not in preference keys."

    def _get_all_actions(self):
        """
        :return: Dictionary containing available actions for each player
        """
        actions = {}
        for player in self.preferences.keys():
            if player == self.my_name:
                continue
            actions[player] = copy.deepcopy(set())

        for joint_action in self.joint_actions_outcomes_mapping.keys():
            self.ego_actions.add(joint_action[self.my_name])

        for player in actions.keys():
            if player == self.my_name:
                continue
            for joint_action in self.joint_actions_outcomes_mapping.keys():
                actions[player].add(joint_action[player])

        return actions

    def _aggregate_outcome(self,
                           action: Trajectory,
                           player_name: PlayerName,
                           joint_actions_outcomes: Mapping[JointTrajectories, JointPlayerOutcome],
                           method: str = "uniform_avg"):
        """
        Compute an aggregated outcome from a set of possible outcomes. The set of outcomes is given by the outcomes of
        all the different joint trajectories than contain a certain action.

        :param action: Action that leads to outcomes
        :param player_name: Player to compute aggregated outcome for
        :param joint_actions_outcomes: Map between joint trajectories and joint outcomes
        :param method: How to aggregate outcomes. Default: "uniform_avg", simply takes arithmetic mean.
        :return: Aggregated outcome for a player and an action
        """

        # list all outcomes for player_name when playing action
        player_outcomes = []
        for joint_traj, joint_outcome in joint_actions_outcomes.items():
            if joint_traj[player_name] == action:
                player_outcomes.append(joint_outcome[player_name])

        # aggregate outcomes by some method
        n_outcomes = len(player_outcomes)
        assert n_outcomes > 0, "There are no outcomes to aggregate"
        aggregation: PlayerOutcome = dict.fromkeys(player_outcomes[0].keys(),
                                                   EvaluatedMetric(name=MetricNodeName("None"), value=-9999.0))

        if method == "uniform_avg":
            for metric in aggregation.keys():
                val = 0.0
                for outcome in player_outcomes:
                    val += outcome[metric].value / n_outcomes
                aggregation[metric] = EvaluatedMetric(name=metric.get_name(), value=val)

        else:
            raise NotImplementedError

        return aggregation

    def _aggregate_all_outcomes(self, method="uniform_avg"):
        """
        Aggregate all outcomes for all players/{my_name}.
        :param method: How to aggregate outcomes. Default: "uniform_avg", simply takes arithmetic mean.
        """
        assert self.preferences.keys() == list(self.joint_actions_outcomes_mapping.keys())[0].keys(), \
            "Players in preferences and joint trajectories don't match."

        # retrieve all available actions for each player from joint actions and joint outcomes dict
        actions = self._get_all_actions()

        action_outcomes_mapping: Mapping[PlayerName, Mapping[Trajectory, PlayerOutcome]] = {}

        for player in actions.keys():
            if player == self.my_name:
                continue
            player_action_outcome_dict: Mapping[Trajectory, PlayerOutcome] = {}
            for action in actions[player]:
                player_action_outcome_dict[action] = \
                    self._aggregate_outcome(action=action, player_name=player,
                                            joint_actions_outcomes=self.joint_actions_outcomes_mapping,
                                            method=method)

            action_outcomes_mapping[player] = deepcopy(player_action_outcome_dict)
        self.aggregated_outcomes = action_outcomes_mapping
        return action_outcomes_mapping

    def _not_second_preferred(self, player: PlayerName, player_outcomes: Mapping[Trajectory, PlayerOutcome]):
        """
        Computes actions that are not "second preferred" to any other action.
        :param player: player for which to remove second_preferred actions
        :param player_outcomes: mapping between (aggregated) actions and outcomes for player
        :return: Set of actions that are not second preferred to any other action
        """
        player_pref = self.preferences[player]
        all_player_actions = set(player_outcomes.keys())

        for traj_1, traj_2 in combinations(player_outcomes.keys(), 2):
            if traj_1 in all_player_actions and traj_2 in all_player_actions:
                result = player_pref.compare(player_outcomes[traj_1], player_outcomes[traj_2])
                if result == FIRST_PREFERRED:
                    all_player_actions.remove(traj_2)
                elif result == SECOND_PREFERRED:
                    all_player_actions.remove(traj_1)
            else:
                continue

        return frozenset(all_player_actions)

    def _player_prior(self, player: PlayerName,
                      player_outcomes: Mapping[Trajectory, PlayerOutcome],
                      method="uniform"):
        """
        Computing a prior for the probability of each action for a player.
        :param player: Player for which to compute prior
        :param player_outcomes: Outcomes (aggregated) for each action available to the player
        :param method: Ho to compute probabilities. Default: "uniform"
        :return:
        """
        actions_never_second_pref = self._not_second_preferred(player, player_outcomes)

        if method == "uniform" or method == "unif":
            n_actions = len(actions_never_second_pref)
            p = {}
            for action in actions_never_second_pref:
                p[action] = Fraction(1, n_actions)

            return ProbDist(p)

        else:
            raise NotImplementedError

    def _compute_all_priors(self):
        """
        Compute priors for all players/{my_name}
        :return:
        """
        action_probs = {}
        outcomes = self._aggregate_all_outcomes()
        for player in self.preferences.keys():
            if player == self.my_name:
                continue
            player_dist = self._player_prior(player=player, player_outcomes=outcomes[player])
            action_probs[player] = player_dist

        return action_probs

    def _compute_outcome_distr_ego(self, ego_action: Trajectory):
        """
        Compute a distribution of outcomes for each action of Ego.
        :param ego_action: Action for which to compute outcomes.
        :return:
        """

        def actions_to_outcomes(joint_action):
            return self.joint_actions_outcomes_mapping[joint_action]

        # compute priors of all actions available to other agents
        all_priors = self._compute_all_priors()

        ego_distr = ProbDist({ego_action: Fraction(1, 1)})

        # include distribution of ego agent in all priors
        all_priors[self.my_name] = ego_distr

        # compute joint action distribution (assuming independence)
        joint_distr = get_joint_distribution_independent(all_priors)

        # convert distribution over actions to distribution over outcomes
        joint_outcome_distr = variable_change(joint_distr, actions_to_outcomes)

        # return only outcome for player "my_name"
        p_ego = {}
        for outcome in joint_outcome_distr.support():
            ego_outcome = outcome[self.my_name]
            if ego_outcome in p_ego.keys():
                p_ego[ego_outcome] += joint_outcome_distr.get(outcome)
            else:
                p_ego[ego_outcome] = joint_outcome_distr.get(outcome)

        return ProbDist(p_ego)

    def get_ego_outcome_distributions(self):
        """
        :return: Distribution over outcomes for all available actions to player "my_name"
        """
        ego_outcome_distr = {}
        for action in self.ego_actions:
            ego_outcome_distr[action] = self._compute_outcome_distr_ego(action)
        return ego_outcome_distr

    def action_selector(self, method: str):
        """
        Select an action from a set of actions where for each action, a distribution over outcomes is given.

        :param method:  method for action selection from distributions of outcomes
                        avg: compare the average outcome for each distribution
                        argmax: compares the most likely outcome for each distribution
                        stochastic_dominance: compare distributions by stochastic dominance:
                                              https://en.wikipedia.org/wiki/Stochastic_dominance

        :return:        best action(s) according to method.
        """
        outcome_distr_ego = self.get_ego_outcome_distributions()

        if method == "avg" or method == "average":
            ego_prob_pref_avg = ProbPrefExpectedValueMetricsDict(p0=self.preferences[self.my_name])
            not_second_pref_actions = never_second_preferred_stochastic(ego_prob_pref_avg, outcome_distr_ego)

        elif method == "argmax":
            ego_prob_pref_argmax = ProbPrefMostLikely(p0=self.preferences[self.my_name])
            not_second_pref_actions = never_second_preferred_stochastic(ego_prob_pref_argmax, outcome_distr_ego)

        elif method == "stochastic_dominance":
            raise NotImplementedError
        else:
            raise NotImplementedError

        return not_second_pref_actions
