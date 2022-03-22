import random
import time
from functools import partial
from itertools import product
from typing import Optional, Mapping, Dict, List, FrozenSet, Set

from dg_commons import U, PlayerName, logger, iterate_dict_combinations, DgSampledSequence
from dg_commons.planning import RefLaneGoal
from dg_commons.planning.trajectory import Trajectory, TrajectoryGraph
from dg_commons.sim import DrawableTrajectoryType
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.vehicle import VehicleState, VehicleCommands
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.scenarios import DgScenario
from dg_commons.sim.simulator_structures import SimObservations
from dg_commons.time import time_function
from driving_games.metrics_structures import PlayerEvaluatedMetrics
from games import MonadicPreferenceBuilder, PURE_STRATEGIES, BAIL_MNE
from possibilities import PossibilitySet
from preferences import SetPreference, Preference
from trajectory_games import TrajectoryGame, SolvingContext, Solution, \
    SolvedTrajectoryGame, TrajectoryGamePlayer, PosetalPreference, TrajectoryGenParams, \
    TrajectoryWorld, MetricEvaluation, TrajGameVisualization, StaticSolverParams, SolvedGameNode, TrajectoryGenerator

__all__ = ["GamePlayingAgent"]


class GamePlayingAgent(Agent):
    """Agent solving a trajectory game"""

    def __init__(
            self,
            map_name: str,
            dg_scenario: DgScenario,
            initial_states: Mapping[PlayerName, VehicleState],
            ref_lanes: Mapping[PlayerName, RefLaneGoal],
            pref_structures: Mapping[PlayerName, str],
            traj_gen_params: Mapping[PlayerName, TrajectoryGenParams],
    ):

        self.map_name = map_name
        self.scenario = dg_scenario
        self.initial_states = initial_states
        self.ref_lanes = ref_lanes
        self.pref_structures = pref_structures
        self.params = traj_gen_params

        self.my_name: Optional[PlayerName] = None
        self.nash_eq: Mapping[str, SolvedTrajectoryGame] = {}
        self.all_trajectories: Mapping[PlayerName, FrozenSet[Trajectory]] = {}
        self.selected_eq: Optional[SolvedGameNode] = None
        self.commands: Optional[DgSampledSequence[VehicleCommands]] = None
        self.trajectory: Optional[Trajectory] = None

    # returns trajectories that are stored in the trajectory graph
    @staticmethod
    def all_trajs_from_graph(player_name: PlayerName,
                             traj_graphs: Mapping[PlayerName, FrozenSet[TrajectoryGraph]]) -> FrozenSet[Trajectory]:

        all_trajectories: Set[Trajectory] = set()
        for graph in traj_graphs[player_name]:
            all_trajectories |= graph.get_all_trajectories_new()
        return frozenset(all_trajectories)

    # return the vehicle commands needed to follow a certain trajectory
    def commands_from_trajectory(self,
                                 traj_graph: Mapping[PlayerName, FrozenSet[TrajectoryGraph]],
                                 trajectory: Trajectory) -> DgSampledSequence:
        # works only for one trajectory graph for each player
        graph = list(traj_graph[self.my_name])[0]

        values = trajectory.values
        timestamps = trajectory.timestamps

        source = (timestamps[0], values[0])
        target = (timestamps[-1], values[-1])
        commands = graph.get_commands_through_nodes(source, target)
        return commands

    @time_function
    def get_traj_game(self,
                      map_name: str,
                      dg_scenario: DgScenario,
                      initial_states: Mapping[PlayerName, VehicleState],
                      ref_lanes: Mapping[PlayerName, RefLaneGoal],
                      pref_structures: Mapping[PlayerName, str],
                      traj_gen_params: Mapping[PlayerName, TrajectoryGenParams]
                      ) -> TrajectoryGame:

        geometries: Dict[PlayerName, VehicleGeometry] = {}
        players: Dict[PlayerName, TrajectoryGamePlayer] = {}
        goals: Dict[PlayerName, List[RefLaneGoal]] = {}

        logger.info(f"Start game generation for GamePlayingAgent")

        ps = PossibilitySet()
        mpref_build: MonadicPreferenceBuilder = SetPreference

        for pname in initial_states.keys():
            initial_state = initial_states[pname]
            pref = PosetalPreference(pref_str=pref_structures[pname], use_cache=False)
            ref_lane_goals = [ref_lanes[pname]]
            goals[pname] = ref_lane_goals
            geometries[pname] = VehicleGeometry.default_car()

            traj_gen = TrajectoryGenerator(params=traj_gen_params[pname], ref_lane_goals=ref_lane_goals)

            players[pname] = TrajectoryGamePlayer(
                name=pname,
                state=ps.unit(initial_state),
                actions_generator=traj_gen,
                preference=pref,
                monadic_preference_builder=mpref_build,
                vg=geometries[pname],
            )

        world = TrajectoryWorld(map_name=map_name, scenario=dg_scenario, geo=geometries, goals=goals)
        get_outcomes = partial(MetricEvaluation.evaluate, world=world)

        game = TrajectoryGame(
            world=world,
            game_players=players,
            ps=ps,
            get_outcomes=get_outcomes,
            game_vis=TrajGameVisualization(world=world, plot_limits="auto"),
        )

        logger.info(f"Game generated.")
        return game

    @staticmethod
    def generate_trajectory_graphs(game: TrajectoryGame, return_graphs: bool) \
            -> Mapping[PlayerName, FrozenSet[TrajectoryGraph]]:

        """Generate graph of trajectories and commands for each player (i.e. get the available actions)"""
        logger.info(f"Generating Trajectories")
        traj_graphs: Mapping[PlayerName, FrozenSet[TrajectoryGraph]] = {}
        for player_name, game_player in game.game_players.items():
            if isinstance(game_player.actions_generator, TrajectoryGenerator):
                states = game_player.state.support()
                assert len(states) == 1, states
                traj_graphs[player_name] \
                    = game_player.actions_generator.get_actions(state=list(states)[0], return_graphs=return_graphs)
            else:
                raise RuntimeError("No trajectory generator found for " + str(player_name))
        return traj_graphs

    def preprocess_game(self,
                        game: TrajectoryGame,
                        traj_graphs: Mapping[PlayerName, FrozenSet[TrajectoryGraph]]) -> SolvingContext:

        def get_context(sgame: TrajectoryGame,
                        actions: Mapping[PlayerName, FrozenSet[Trajectory]]) -> SolvingContext:

            pref: Mapping[PlayerName, Preference[PlayerEvaluatedMetrics]] = {
                name: player.preference for name, player in sgame.game_players.items()
            }

            # todo[LEON]: currently not used, remove
            solver_params = StaticSolverParams(
                admissible_strategies=PURE_STRATEGIES,
                strategy_multiple_nash=BAIL_MNE,
                dt=1.,
                factorization_algorithm="TEST",
                use_factorization=False,
                n_simulations=5,
                extra=False,
                max_depth=3
            )
            kwargs = {
                "player_actions": actions,
                "game_outcomes": sgame.get_outcomes,
                "outcome_pref": pref,
                "solver_params": solver_params,
            }

            return SolvingContext(**kwargs)

        all_trajectories: Mapping[PlayerName, FrozenSet[Trajectory]] = {}

        for player_name, game_player in game.game_players.items():
            all_trajectories[player_name] = self.all_trajs_from_graph(player_name=player_name, traj_graphs=traj_graphs)

        for joint_traj in set(iterate_dict_combinations(all_trajectories)):
            game.get_outcomes(joint_traj)

        self.all_trajectories = all_trajectories

        return get_context(sgame=game, actions=all_trajectories)

    def select_admissible_eq_random(self):
        admissible_eq = list(self.nash_eq['admissible'])
        if len(admissible_eq) > 1:
            logger.info(
                f"Randomly selecting one admissible equilibria out of "
                f"{len(admissible_eq)}" f"available equilibria."
            )
        return random.choice(admissible_eq)

    @time_function
    def on_episode_init(self, my_name: PlayerName):
        self.my_name = my_name
        game = self.get_traj_game(
            map_name=self.map_name,
            dg_scenario=self.scenario,
            initial_states=self.initial_states,
            ref_lanes=self.ref_lanes,
            pref_structures=self.pref_structures,
            traj_gen_params=self.params
        )
        traj_graphs: Mapping[PlayerName, FrozenSet[TrajectoryGraph]] = \
            self.generate_trajectory_graphs(game, return_graphs=True)
        solving_context: SolvingContext = self.preprocess_game(game, traj_graphs=traj_graphs)
        sol: Solution = Solution()
        self.nash_eq: Mapping[str, SolvedTrajectoryGame] = sol.solve_game(context=solving_context)
        self.selected_eq = self.select_admissible_eq_random()  # randomly select one of the admissible equilibria
        self.trajectory = self.selected_eq.actions[self.my_name]
        self.commands = self.commands_from_trajectory(traj_graph=traj_graphs, trajectory=self.trajectory)


    def get_commands(self, sim_obs: SimObservations) -> U:
        current_time = sim_obs.time
        if current_time < self.commands.get_start():
            logger.info('Warning, no commands defined so early. Returning first command input.')
            return self.commands.values[0]
        elif current_time > self.commands.get_end():
            logger.info('Warning, no commands defined so late. Returning last command input.')
            return self.commands.values[-1]
        else:
            return self.commands.at_or_previous(current_time) #todo: strange: at_interp is better at following a trajectory (only in one case)

    def on_get_extra(self) -> Optional[DrawableTrajectoryType]:
        trajectories = self.all_trajectories[self.my_name]
        trajectories_blue = self.all_trajectories[PlayerName('P1')]
        selected_traj = self.trajectory
        candidates = tuple(
            product(
                trajectories,
                [
                    "lightcoral",
                ],
            )
        )
        new_tuple = (selected_traj, 'red')
        candidates += (new_tuple,)

        candidates_blue = tuple(
            product(
                trajectories_blue,
                [
                    "blue",
                ],
            )
        )
        candidates += candidates_blue

        return candidates
