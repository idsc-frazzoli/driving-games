import pomdp_py
from models import PolicyModel, EgoTransition, ObservationModel, RewardModel
from pomdp_py.utils import TreeDebugger


# agent and environment are here for now. Of course we will need to define something more interesting.
class LeonsProblem(pomdp_py.POMDP):

    def __init__(self, init_true_state=None, init_belief=None, dt=0.01):
        """init_belief is a Distribution."""
        agent = pomdp_py.Agent(init_belief,
                               policy_model=PolicyModel(),
                               transition_model=EgoTransition(dt=dt),
                               observation_model=ObservationModel(),
                               reward_model=RewardModel())
        env = pomdp_py.Environment(init_true_state,
                                   EgoTransition(),
                                   RewardModel())
        super().__init__(agent, env, name="LeonsProblem")


def test_planner(problem, planner, nsteps=3, debug_tree=False):
    """
    Runs the action-feedback loop of Tiger problem POMDP

    Args:
        problem (TigerProblem): an instance of the tiger problem.
        planner (Planner): a planner
        nsteps (int): Maximum number of steps to run this loop.
    """
    for i in range(nsteps):
        action = planner.plan(problem.agent)
        if debug_tree:
            from pomdp_py.utils import TreeDebugger
            dd = TreeDebugger(problem.agent.tree)
            import pdb; pdb.set_trace()

        print("==== Step %d ====" % (i+1))
        print("True state: %s" % problem.env.state)
        print("Belief: %s" % str(problem.agent.cur_belief))
        print("Action: %s" % str(action))
        print("Reward: %s" % str(problem.env.reward_model.sample(problem.env.state, action, None)))

        # Let's create some simulated real observation; Update the belief
        # Creating true observation for sanity checking solver behavior.
        # In general, this observation should be sampled from agent's observation model.
        #real_observation = ObservationModel(problem.env.state.name)
        real_observation = ObservationModel()
        print(">> Observation: %s" % real_observation)
        problem.agent.update_history(action, real_observation)

        # If the planner is POMCP, planner.update also updates agent belief.
        planner.update(problem.agent, action, real_observation)
        if isinstance(planner, pomdp_py.POUCT):
            print("Num sims: %d" % planner.last_num_sims)
            print("Plan time: %.5f" % planner.last_planning_time)

        if isinstance(problem.agent.cur_belief, pomdp_py.Histogram):
            new_belief = pomdp_py.update_histogram_belief(problem.agent.cur_belief,
                                                          action, real_observation,
                                                          problem.agent.observation_model,
                                                          problem.agent.transition_model)
            problem.agent.set_belief(new_belief)

        if action.name.startswith("open"):
            # Make it clearer to see what actions are taken until every time door is opened.
            print("\n")



def main():
    init_true_state = 0
    init_belief = 0
    leons_problem = LeonsProblem(init_true_state=init_true_state, init_belief=init_belief, dt=0.01)

    # planner POMCP
    pomcp = pomdp_py.POMCP(max_depth=3, discount_factor=0.95,
                           planning_time=.5, exploration_const=110,
                           rollout_policy=leons_problem.agent.policy_model)

    print("** Testing POMCP **")
    leons_problem.agent.set_belief(pomdp_py.Particles.from_histogram(init_belief, num_particles=100), prior=True)
    pomcp = pomdp_py.POMCP(max_depth=3, discount_factor=0.95,
                           num_sims=1000, exploration_const=50,
                           rollout_policy=leons_problem.agent.policy_model,
                           show_progress=True, pbar_update_interval=500)
    test_planner(leons_problem, pomcp, nsteps=10)
    TreeDebugger(leons_problem.agent.tree).pp


if __name__ == '__main__':
    main()
