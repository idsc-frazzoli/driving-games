from functools import partial
from typing import Mapping, FrozenSet, TypeVar, List

from dg_commons import PlayerName, iterate_dict_combinations
from dg_commons.planning import Trajectory, JointTrajectories
from driving_games.metrics_structures import JointPlayerOutcome, PlayerOutcome, Metric
from possibilities import ProbDist
from preferences import Preference
from trajectory_games import TrajectoryWorld, MetricEvaluation
from trajectory_games.metrics import DrivableAreaViolation

P = TypeVar("P")


class UncertainActionEstimator:

    def __init__(self, world: TrajectoryWorld, actions: Mapping[PlayerName, FrozenSet[Trajectory]]):
        self.world: TrajectoryWorld = world
        self.actions: Mapping[PlayerName, FrozenSet[Trajectory]] = actions
        # self.action_probabilities: Mapping[PlayerName, ProbDist[Trajectory]] = {}
        # todo: implement preferences instead of just checking least preferred like now
        # self.preferences: Mapping[PlayerName, Preference[P]] = {}
        self.outcomes_dict: Mapping[JointTrajectories, JointPlayerOutcome] = {}
        # assert self.actions.keys() == self.action_probabilities.keys()
        # assert self.actions.keys() == self.preferences.keys()
        pr
        self.compute_outcomes()
        a=10

    def compute_outcomes(self):
        get_outcomes = partial(MetricEvaluation.evaluate, world=self.world)
        outcomes_dict: Mapping[JointTrajectories, JointPlayerOutcome] = {}
        for joint_action in iterate_dict_combinations(self.actions):
            outcomes_dict[joint_action] = get_outcomes(joint_action)

        self.outcomes_dict = outcomes_dict

    def _action_player_outcome(self, action: Trajectory, player_name: PlayerName):
        player_outcomes = []
        for joint_traj, joint_outcome in self.outcomes_dict.items():
            if action == joint_traj[player_name]:
                player_outcomes.append(joint_outcome[player_name])

        return self.aggregate_outcome(player_outcomes, method="expected_value")

    def aggregate_outcome(self, outcomes: List[PlayerOutcome], method: str = "expected_value") -> PlayerOutcome:
        assert len(outcomes) > 0, "There are no outcomes to aggregate"
        aggregation: PlayerOutcome = outcomes[0]
        for metric in aggregation.keys():
            aggregation[metric].value = 0.0

        n_outcomes = len(outcomes)

        if method == "expected_value":
            for outcome in outcomes:
                for metric in outcome.keys():
                    aggregation[metric].value += outcome[metric].value / n_outcomes
        else:
            raise NotImplementedError

        return aggregation

    def player_outcomes(self, player_name: PlayerName):
        player_outcomes: Mapping[Trajectory, PlayerOutcome] = {}
        for action in self.actions[player_name]:
            agg_outcome = self._action_player_outcome(action=action, player_name=player_name)
            player_outcomes[action] = agg_outcome

        return player_outcomes

    def smallest_values_single_metric(self, player_outcomes: Mapping[Trajectory, PlayerOutcome],
                                      metric: Metric, tol: float = 0.1):
        best_actions = []
        single_metric_outcome = {}
        for traj, outcome in player_outcomes.items():
            single_metric_outcome[traj] = outcome[metric].value

        min_value = min(single_metric_outcome.values())
        for traj, value in single_metric_outcome.values():
            if abs(min_value - value) < tol:
                best_actions.append(traj)

        return best_actions

    def compute_player_probabilities(self, player_name: PlayerName):
        player_outcomes = self.player_outcomes(player_name)
        # todo: ranked by a single metric: generalize
        best_actions = self.smallest_values_single_metric(player_outcomes=player_outcomes,
                                                          metric=DrivableAreaViolation())
        print(best_actions)
        a = 10
