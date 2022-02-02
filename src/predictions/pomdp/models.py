import random

import pomdp_py
import numpy as np

from domain import OtherObservation, EgoObservation, EgoState, ExtendedState, MotionAction


# Transition model
class EgoTransition(pomdp_py.TransitionModel):
    def __init__(self, dt):
        self.A = np.array([[1., dt, 0.], [0., 1., 0.], [0., 0., 1.]])
        self.b = np.array([[0.5 * dt * dt], [dt], [0.]])

    # deterministic
    def sample(self, state, action):
        #return np.matmul(self.A, state) + self.b * action
        return state + action

    def probability(self, next_state, state, action):
        raise NotImplementedError

    def get_distribution(self, state, action):
        raise NotImplementedError

    def get_all_states(self):
        raise NotImplementedError

    def argmax(self, state, action):
        raise NotImplementedError


# Policy Model
class PolicyModel(pomdp_py.RolloutPolicy):
    """Just return an acceleration randomly"""

    def sample(self, state):
        raise NotImplementedError

    def probability(self, action, state, **kwargs):
        raise NotImplementedError

    def argmax(self, state, **kwargs):
        """Returns the most likely action"""
        raise NotImplementedError

    def get_all_actions(self):
        raise NotImplementedError

    def rollout(self):
        return random.sample([-1., 0., 1.])

# Observation model
class ObservationModel(pomdp_py.ObservationModel):
    #def __init__(self):

    def probability(self, observation, next_state, action):
        """ Returns the probability Pr(o | s’,a). """
        print("probability of observation model called.")
        # return every route as equally likely for now
        return 1./3

    def sample(self, next_state, action):
        """ Returns a sample o ~ Pr(o | s’,a). """
        print("sample of observation model called")
        # return a random route in [1,2,3] with uniform probability
        return random.sample([1, 2, 3], 1)

    def argmax(self, next_state, action):
        print("argmax of bservation model")
        return 1

    def get_all_observations(self):
        print("get_all_observations called")
        return [1, 2, 3]

# Reward model
class RewardModel(pomdp_py.RewardModel):
    """
        A RewardModel models the distribution :math:`\Pr(r|s,a,s')` where
            :math:`r\in\mathbb{R}` with `argmax` denoted as denoted as
            :math:`R(s,a,s')`.
        """

    def probability(self, reward, state, action, next_state):
        raise NotImplementedError

    def sample(self, state, action, next_state):
        return 100

    def argmax(self, state, action, next_state):
        raise NotImplementedError