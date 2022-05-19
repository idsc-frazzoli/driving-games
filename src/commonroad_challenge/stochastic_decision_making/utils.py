import random
from copy import deepcopy
from fractions import Fraction
from functools import partial
from typing import Mapping, FrozenSet, TypeVar, List

from dg_commons import PlayerName, iterate_dict_combinations
from dg_commons.planning import Trajectory, JointTrajectories
from driving_games.metrics_structures import JointPlayerOutcome, PlayerOutcome, Metric, EvaluatedMetric
from possibilities import ProbDist
from preferences import Preference, SECOND_PREFERRED, FIRST_PREFERRED, INCOMPARABLE, INDIFFERENT
from trajectory_games import TrajectoryWorld, MetricEvaluation
from trajectory_games.metrics import DrivableAreaViolation, LongitudinalAccelerationSquared

P = TypeVar("P")
PlayerOutcomeFloat = Mapping[Metric, float]


# todo:
# estimator takes: actions, preferences, world for evaluation
# estimator returns: distributions over outcomes for each action of "EGO".

class ActionLikelihoodEstimator:

    def __init__(
            self,
            # joint_actions_outcomes_dict: Mapping[JointTrajectories, JointPlayerOutcome],
            # give as argument to function, not in init
            preferences: Mapping[PlayerName, Preference[P]]

    ):

        # self.joint_actions_outcome_dict = joint_actions_outcomes_dict
        self.preferences = preferences
        # self.actions: Mapping[PlayerName, FrozenSet[Trajectory]] = {}
        # self.actions_outcomes_dict: Mapping[
        #     PlayerName, Mapping[Trajectory, PlayerOutcomeFloat]] = self.get_actions_to_outcomes_dict(
        #     joint_actions_outcomes_dict)
        # self.get_actions_dict()

    # BLOCK 1

    def get_actions_to_outcomes_dict(self, joint_actions_outcomes: Mapping[JointTrajectories, JointPlayerOutcome]):
        a = list(joint_actions_outcomes.keys())[0].keys()
        b = self.preferences.keys()
        assert self.preferences.keys() == list(joint_actions_outcomes.keys())[0].keys(), "Players in preferences " \
                                                                                         "and joint trajectories don't" \
                                                                                         "match."
        # retrieve all actions for each player from joint actions and joint outcomes dict
        actions = dict.fromkeys(self.preferences.keys(), set())
        for player in actions.keys():
            for joint_action in joint_actions_outcomes.keys():
                actions[player].add(joint_action[player])

        action_outcomes_mapping: Mapping[PlayerName, Mapping[Trajectory, PlayerOutcomeFloat]] = {}

        for player in actions.keys():
            player_action_outcome_dict: Mapping[Trajectory, PlayerOutcomeFloat] = {}
            for action in actions[player]:
                player_action_outcome_dict[action] = self._player_action_outcome(action,
                                                                                 player,
                                                                                 joint_actions_outcomes,
                                                                                 "uniform_avg")

            action_outcomes_mapping[player] = deepcopy(player_action_outcome_dict)

        # player_outcomes: Mapping[Trajectory, PlayerOutcomeFloat] = {}
        # for action in self.actions[player_name]:
        #     agg_outcome = self._player_action_outcome(action=action, player_name=player_name)
        #     player_outcomes[action] = agg_outcome

        return action_outcomes_mapping

    def _player_action_outcome(self,
                               action: Trajectory,
                               player_name: PlayerName,
                               joint_actions_outcomes: Mapping[JointTrajectories, JointPlayerOutcome],
                               method: str ="uniform_avg"):
        player_outcomes = []
        for joint_traj, joint_outcome in joint_actions_outcomes.items():
            if joint_traj[player_name] == action:
                player_outcomes.append(joint_outcome[player_name])

        return self._aggregate_outcome(player_outcomes, method=method)

    def _aggregate_outcome(self, outcomes: List[PlayerOutcome], method: str = "uniform_avg") -> PlayerOutcomeFloat:
        assert len(outcomes) > 0, "There are no outcomes to aggregate"
        aggregation: PlayerOutcomeFloat = dict.fromkeys(outcomes[0].keys(), 0.0)
        n_outcomes = len(outcomes)

        if method == "uniform_avg":
            for outcome in outcomes:
                for metric in aggregation.keys():
                    aggregation[metric] += outcome[metric].value / n_outcomes
        else:
            raise NotImplementedError
        return aggregation

    # def player_outcomes(self, player_name: PlayerName) -> Mapping[Trajectory, PlayerOutcomeFloat]:
    #     player_outcomes: Mapping[Trajectory, PlayerOutcomeFloat] = {}
    #     for action in self.actions[player_name]:
    #         agg_outcome = self._player_action_outcome(action=action, player_name=player_name)
    #         player_outcomes[action] = agg_outcome
    #
    #     return player_outcomes

    # def get_actions_dict(self):
    #     self.actions = dict.fromkeys(self.preferences.keys(), set())
    #     for player in self.actions.keys():
    #         for joint_action in self.joint_actions_outcome_dict.keys():
    #             self.actions[player].add(joint_action[player])


    # # BLOCK 2
    # def _player_action_prob(self, player: PlayerName) -> ProbDist[Trajectory]:
    #     ranked_actions = []
    #     player_pref = self.preferences[player]
    #     other_action = None
    #     # todo: use outcomes not actions
    #     for action in self.actions[player]:
    #         # insert first action
    #         if not ranked_actions:
    #             ranked_actions.append(action)
    #             other_action = ranked_actions[-1]
    #             continue
    #
    #         concluded = False
    #         while not concluded:
    #
    #             if player_pref.compare(action, other_action) == SECOND_PREFERRED:
    #                 ranked_actions.append(action)
    #                 other_action = action
    #                 concluded = True
    #
    #             elif player_pref.compare(action, other_action) == FIRST_PREFERRED:
    #                 idx_other = ranked_actions.index(other_action)
    #                 other_action = ranked_actions[idx_other - 1]
    #                 concluded = False
    #
    #             elif player_pref.compare(action, other_action) == INDIFFERENT:
    #                 # insert before or after with probability 0.5
    #                 idx_other = ranked_actions.index(other_action)
    #                 if random.random() < 0.5:
    #                     ranked_actions.insert(idx_other, action)
    #                 else:
    #                     ranked_actions.insert(idx_other + 1, action)
    #                 concluded = True
    #
    #
    #             elif player_pref.compare(action, other_action) == INCOMPARABLE:
    #                 # todo: for now just consider total orders, ignore incomparable cases
    #                 continue
    #
    #     pass
    #
    # def action_probs(self) -> Mapping[PlayerName, ProbDist[Trajectory]]:
    #     probs = {}
    #     for player in self.preferences.keys():
    #         probs[player] = self._player_action_prob(player)
    #     return probs


class UncertainActionEstimator:

    def __init__(
            self,
            world: TrajectoryWorld,
            actions: Mapping[PlayerName, FrozenSet[Trajectory]],
            preferences: Mapping[PlayerName, Preference[P]]
    ):
        self.world: TrajectoryWorld = world
        self.actions: Mapping[PlayerName, FrozenSet[Trajectory]] = actions
        # self.action_probabilities: Mapping[PlayerName, ProbDist[Trajectory]] = {}
        # todo: implement preferences instead of just checking least preferred like now
        self.preferences: Mapping[PlayerName, Preference[P]] = preferences
        self.outcomes_dict: Mapping[JointTrajectories, JointPlayerOutcome] = {}
        # assert self.actions.keys() == self.action_probabilities.keys()
        # assert self.actions.keys() == self.preferences.keys()
        self.compute_outcomes()

    def compute_outcomes(self):
        get_outcomes = partial(MetricEvaluation.evaluate, world=self.world)
        outcomes_dict: Mapping[JointTrajectories, JointPlayerOutcome] = {}
        for joint_action in iterate_dict_combinations(self.actions):
            outcomes_dict[joint_action] = get_outcomes(joint_action)
            print(outcomes_dict)

        self.outcomes_dict = outcomes_dict

    #
    # def _action_player_outcome(self, action: Trajectory, player_name: PlayerName):
    #     player_outcomes = []
    #     for joint_traj, joint_outcome in self.outcomes_dict.items():
    #         if action == joint_traj[player_name]:
    #             player_outcomes.append(joint_outcome[player_name])
    #
    #     return self._aggregate_outcome(player_outcomes, method="expected_value")

    # def _aggregate_outcome(self, outcomes: List[PlayerOutcome], method: str = "expected_value") -> PlayerOutcomeFloat:
    #     assert len(outcomes) > 0, "There are no outcomes to aggregate"
    #     aggregation: PlayerOutcomeFloat = dict.fromkeys(outcomes[0].keys(), 0.0)
    #
    #     n_outcomes = len(outcomes)
    #
    #     if method == "expected_value":
    #         for outcome in outcomes:
    #             for metric in aggregation.keys():
    #                 aggregation[metric] += outcome[metric].value / n_outcomes
    #     else:
    #         raise NotImplementedError
    #
    #     return aggregation

    # def player_outcomes(self, player_name: PlayerName):
    #     player_outcomes: Mapping[Trajectory, PlayerOutcome] = {}
    #     for action in self.actions[player_name]:
    #         agg_outcome = self._action_player_outcome(action=action, player_name=player_name)
    #         player_outcomes[action] = agg_outcome
    #
    #     return player_outcomes

    def smallest_values_single_metric(self, player_outcomes: Mapping[Trajectory, PlayerOutcomeFloat],
                                      metric: Metric, tol: float = 0.1):
        best_actions = []
        single_metric_outcome = {}
        for traj, outcome in player_outcomes.items():
            single_metric_outcome[traj] = outcome[metric]

        min_value = min(single_metric_outcome.values())
        for traj, value in single_metric_outcome.items():
            if abs(min_value - value) < tol:
                best_actions.append(traj)

        return best_actions

    def compute_player_probabilities(self, player_name: PlayerName):
        player_outcomes = self.player_outcomes(player_name)
        # todo: ranked by a single metric: generalize
        best_actions = self.smallest_values_single_metric(player_outcomes=player_outcomes,
                                                          metric=LongitudinalAccelerationSquared())
        # print(best_actions)

        distr = ProbDist(p={best_actions[0]: Fraction(1, 1)})
        print(distr.support())
        return distr

    def compute_outcome_distribution(self, player_name: PlayerName) -> Mapping[
        Trajectory, ProbDist[PlayerOutcomeFloat]]:
        """
        Compute outcome distribution by computing distributions over actions for all players
        :param player_name: player for which to compute outcome distribution
        :return: Dictionary mapping each action on player_name to a distribution over outcomes
        """

        players = self.world.get_players()
        action_distr: Mapping[PlayerName, ProbDist[Trajectory]] = {}
        for player in players:
            if player != player_name:
                action_distr[player] = self.compute_player_probabilities(player)

        # todo: here need to build probability distribution of joint actions -> simply multiply?
        # todo: is the distribution actually independent?
        # todo: then map this to outcome space, through change of variables

        # func = LongitudinalAccelerationSquared().evaluate... # here use partial. Partially initialize the context on top
        # pass this func to variable change
        from possibilities.prob import variable_change

    def action_selector(self):
        # todo:
        pass
