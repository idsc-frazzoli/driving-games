import pomdp_py
from dg_commons.sim.simulator import SimContext



class FourWayCrossingEnvironment(pomdp_py.Environment):
    """
    An Environment maintains the true state of the world.
        For example, it is the 2D gridworld, rendered by pygame.
        Or it could be the 3D simulated world rendered by OpenGL.
        Therefore, when coding up an Environment, the developer
        should have in mind how to represent the state so that
        it can be used by a POMDP or OOPOMDP.

        The Environment is passive. It never observes nor acts.
    """

    def __init__(self, sim_context: SimContext):
        self.sim_context = sim_context

    def apply_transition(self, next_state):  # real signature unknown; restored from __doc__
        """
        apply_transition(self, next_state)
                Apply the transition, that is, assign current state to
                be the `next_state`.
        """
        pass

    def execute(self, *args, **kwargs):  # real signature unknown
        pass

    def provide_observation(self, observation_model, action):  # real signature unknown; restored from __doc__
        """
        provide_observation(self, observation_model, action)
                Returns an observation sampled according to :math:`\Pr(o|s',a)`
                where :math:`s'` is current environment :meth:`state`, :math:`a`
                is the given `action`, and :math:`\Pr(o|s',a)` is the `observation_model`.

                Args:
                    observation_model (ObservationModel)
                    action (Action)

                Returns:
                    Observation: an observation sampled from :math:`\Pr(o|s',a)`.
        """
        pass

    def state_transition(self, action, execute=True):  # real signature unknown; restored from __doc__
        """
        state_transition(self, action, execute=True)
                Simulates a state transition given `action`. If `execute` is set to True,
                then the resulting state will be the new current state of the environment.

                Args:
                    action (Action): action that triggers the state transition
                    execute (bool): If True, the resulting state of the transition will become the current state.
                    discount_factor (float): Only necessary if action is an Option. It is the discount
                        factor when executing actions following an option's policy until reaching terminal condition.

                Returns:
                    float or tuple: reward as a result of `action` and state transition, if `execute` is True
                    (next_state, reward) if `execute` is False.
        """
        pass