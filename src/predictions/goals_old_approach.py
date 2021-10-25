import os
from datetime import datetime
from typing import Mapping, Dict
from numpy import cos
from numpy import sin

from dg_commons import PlayerName, X
from sim import logger
from sim.simulator import SimContext, Simulator
from sim.simulator_structures import *
from crash_tests.test_scenarios import *
from crash.reports import generate_report



class GoalGenerator(Simulator): #non very elegant naming. Admissible to call simulator inside of another class?
    """
    The goal generator generates possible goals for all agents.
    """
    goals : Mapping[PlayerName, X]

    #for now test framework with simple goal inference model
    def infer_goals(self, sim_context: SimContext):
        for player_name in sim_context.models.items():
            player_state = self.last_observations.players[player_name].state
            x_new = player_state.x + 2*cos(player_state)
            y_new = player_state.y + 2*sin(player_state)
            self.goals[player_name] = [x_new, y_new] #make state?

    def visualize_goals(self, sim_logger: Dict[PlayerName, PlayerLogger] ):
        # this function will receive the point in space found by infer_goals
        # and pass them to extra

    # this should run a simulator and at every timestep make a prediction
    #modified version of run that includes making goals at the end of every loop
    def run(self, sim_context: SimContext):
        logger.info("Beginning simulation.")
        for player_name, player in sim_context.players.items():
            player.on_episode_init(player_name)
            self.simlogger[player_name] = PlayerLogger()
        while not sim_context.sim_terminated:
            self.pre_update(sim_context)
            self.update(sim_context)
            self.infer_goals(sim_context) #goals generated after update
            self.post_update(sim_context)
        logger.info("Completed simulation. Writing logs...")
        for player_name in sim_context.players:
            sim_context.log[player_name] = self.simlogger[player_name].as_sequence()
        logger.info("Writing logs terminated.")

#define when goals are updated. At each timestep? MAkes sense actually
# calculate goals at each timestep. For now just say goal is along a line
# in direction of driving, and make framework to include this in simulation and
# plot the changing goals
# this class should be implemented


sim_context = get_scenario_racetrack_test()
sim_context = get_scenario_suicidal_pedestrian()
run_scenario_without_compmake(sim_context)



def run_simulation(goal_generator : GoalGenerator, sim_context : SimContext) -> SimContext:
    # run simulations
    goal_generator.run(sim_context)
    return sim_context

def run_scenario_without_compmake(sim_context: SimContext, output_dir: str = "out"):
    sim = Simulator()
    # run simulations
    sim.run(sim_context)
    # generate collisions and damages report
    report = generate_report(sim_context)
    # save report
    now_str = datetime.now().strftime("%y-%m-%d-%H%M%S")
    report_file = os.path.join(output_dir, f"optimal_crash_{now_str}.html")
    report.to_html(report_file)