from fractions import Fraction
from functools import partial
from typing import Mapping, FrozenSet, TypeVar, List

from dg_commons import PlayerName, iterate_dict_combinations
from dg_commons.planning import Trajectory, JointTrajectories
from driving_games.metrics_structures import JointPlayerOutcome, PlayerOutcome, Metric, EvaluatedMetric
from possibilities import ProbDist
from preferences import Preference
from trajectory_games import TrajectoryWorld, MetricEvaluation
from trajectory_games.metrics import DrivableAreaViolation, LongitudinalAccelerationSquared

P = TypeVar("P")
PlayerOutcomeFloat = Mapping[Metric, float]


# todo:
# estimator takes: actions, preferences, world for evaluation
# estimator returns: distributions over outcomes for each action of "EGO".

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

    def _action_player_outcome(self, action: Trajectory, player_name: PlayerName):
        player_outcomes = []
        for joint_traj, joint_outcome in self.outcomes_dict.items():
            if action == joint_traj[player_name]:
                player_outcomes.append(joint_outcome[player_name])

        return self.aggregate_outcome(player_outcomes, method="expected_value")

    def aggregate_outcome(self, outcomes: List[PlayerOutcome], method: str = "expected_value") -> PlayerOutcomeFloat:
        assert len(outcomes) > 0, "There are no outcomes to aggregate"
        aggregation: PlayerOutcomeFloat = dict.fromkeys(outcomes[0].keys(), 0.0)

        n_outcomes = len(outcomes)

        if method == "expected_value":
            for outcome in outcomes:
                for metric in aggregation.keys():
                    aggregation[metric] += outcome[metric].value / n_outcomes
        else:
            raise NotImplementedError

        return aggregation

    def player_outcomes(self, player_name: PlayerName):
        player_outcomes: Mapping[Trajectory, PlayerOutcome] = {}
        for action in self.actions[player_name]:
            agg_outcome = self._action_player_outcome(action=action, player_name=player_name)
            player_outcomes[action] = agg_outcome

        return player_outcomes

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

    def compute_outcome_distribution(self, player_name: PlayerName) -> Mapping[Trajectory, ProbDist[PlayerOutcomeFloat]]:
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

