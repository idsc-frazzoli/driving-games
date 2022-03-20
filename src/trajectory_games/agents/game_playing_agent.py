from functools import partial
from typing import Optional, Mapping, Dict, List, FrozenSet, Set
from decimal import Decimal as D

from frozendict import frozendict

from dg_commons import U, PlayerName, logger, iterate_dict_combinations
from dg_commons.planning import RefLaneGoal
from dg_commons.planning.trajectory import Trajectory
from dg_commons.sim import DrawableTrajectoryType
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.scenarios import DgScenario, load_commonroad_scenario
from dg_commons.sim.simulator_structures import SimObservations
from dg_commons.time import time_function
from driving_games.metrics_structures import PlayerEvaluatedMetrics
from games import MonadicPreferenceBuilder, PURE_STRATEGIES, BAIL_MNE
from possibilities import PossibilitySet
from preferences import SetPreference, Preference
from trajectory_games.agents.stop_or_go_agent import *
from trajectory_games import TrajectoryGame, SolvingContext, get_simple_traj_game_leon, preprocess_full_game, Solution, \
    SolvedTrajectoryGame, TrajectoryGamePlayer, PosetalPreference, TrajectoryGenParams, TrajectoryGenerator, \
    TrajectoryWorld, MetricEvaluation, TrajGameVisualization, StaticSolverParams
from trajectory_games.game_def import FrozenTrajectories
from trajectory_games.trajectory_generator import TrajectoryGeneratorCommands
from trajectory_games_tests.test_games import report_single

__all__ = ["GamePlayingAgent"]


class GamePlayingAgent(Agent):
    """Agent solving a trajectory game"""

    def __init__(
            self,
            map_name: str,
            initial_states: Mapping[PlayerName, VehicleState],
            ref_lanes: Mapping[PlayerName, RefLaneGoal],
            pref_structures: Mapping[PlayerName, str],
            transition_gen_map: Mapping[PlayerName, str],
            trajectory_file_path: str ="trajectory_stop_go.pickle",
            stop_or_go_behavior: str = "stop",



            reporting: bool = False):

        # self.traj_gen, self.commands_gen = self.get_stop_or_go_traj(trajectory_file_path=trajectory_file_path,
        #                                                             behavior=stop_or_go_behavior)
        self.game: TrajectoryGame = self.get_traj_game(
            map_name=map_name,
            initial_states=initial_states,
            ref_lanes=ref_lanes,
            pref_structures=pref_structures,
            trajectory_file_path=trajectory_file_path,
            stop_or_go_behavior=stop_or_go_behavior,
            transition_gen_map=transition_gen_map
        )

        self.solving_context = self.preprocess_game(self.game)
        self.sol = Solution()
        self.nash_eq: Mapping[str, SolvedTrajectoryGame] = self.sol.solve_game(context=self.solving_context)
        self.debug = True
        # todo: now we have a flow that computes NASH EQ for ego -> still need to return vehicle commands
        # self.my_name = None
        # self.trajectory = None

        # OLD
        self.reporting = reporting

    @staticmethod
    def get_stop_or_go_traj(trajectory_file_path: str, behavior: str):
        traj_gen = read_traj(trajectory_file_path, behavior)["trajectory"]
        return traj_gen

    @time_function
    def get_traj_game(self, map_name: str,
                      initial_states: Mapping[PlayerName, VehicleState],
                      ref_lanes: Mapping[PlayerName, RefLaneGoal],
                      pref_structures: Mapping[PlayerName, str],
                      trajectory_file_path: str,
                      stop_or_go_behavior: str,
                      transition_gen_map: Mapping[PlayerName, str],
                      stop_or_go_behavior_me_DEV: str = "go", #todo [LEON]: DEV, remove and use trajectory generator
                      ) -> TrajectoryGame:

        geometries: Dict[PlayerName, VehicleGeometry] = {}
        players: Dict[PlayerName, TrajectoryGamePlayer] = {}
        goals: Dict[PlayerName, List[RefLaneGoal]] = {}
        scenario: DgScenario
        logger.info(f"Loading Scenario: {map_name}")
        scenarios_dir = "/home/leon/Documents/repos/driving-games/scenarios"
        scenario, _ = load_commonroad_scenario(map_name, scenarios_dir=scenarios_dir)
        logger.info("Done.")

        logger.info(f"Start game generation for GamePlayingAgent")

        ps = PossibilitySet()
        mpref_build: MonadicPreferenceBuilder = SetPreference

        for pname in initial_states.keys():

            initial_state = initial_states[pname]

            pref = PosetalPreference(pref_str=pref_structures[pname], use_cache=False)

            ref_lane_player = ref_lanes[pname]
            ref_lane_goals = [ref_lane_player]

            # transition generator
            if transition_gen_map[pname] == "generate":
                trans_param = TrajectoryGenParams.default()
                traj_gen = TrajectoryGeneratorCommands(params=trans_param, ref_lane_goals=ref_lane_goals)
            # load precomputed set of trajectories
            elif transition_gen_map[pname] == "stop-or-go":
                if pname == "P1":
                    traj_gen = self.get_stop_or_go_traj(
                        trajectory_file_path=trajectory_file_path,
                        behavior=stop_or_go_behavior
                    )
                elif pname == "Ego":
                    # issues when u_acc <= 0.0
                    u_acc = frozenset([1.0, 2.0])
                    u_dst = frozenset([0.0])
                    # u_dst = frozenset([_ * 0.2 for _ in u_acc])

                    params = TrajectoryGenParams(
                        solve=False,
                        s_final=-1,
                        max_gen=100,
                        dt=D("1.0"),
                        # keep at max 1 sec, increase k_maxgen in trajectrory_generator for having more generations
                        u_acc=u_acc,
                        u_dst=u_dst,
                        v_max=15.0,
                        v_min=0.0,
                        st_max=0.5,
                        dst_max=1.0,
                        dt_samp=D("0.2"),
                        dst_scale=False,
                        n_factor=0.8,
                        vg=VehicleGeometry.default_car(),
                    )
                    traj_gen = TrajectoryGenerator(params=params, ref_lane_goals=ref_lane_goals)
            else:
                raise ValueError("This type of transition generator is not yet defined.")

            goals[pname] = ref_lane_goals

            geometries[pname] = VehicleGeometry.default_car()

            players[pname] = TrajectoryGamePlayer(
                name=pname,
                state=ps.unit(initial_state),
                actions_generator=traj_gen,
                preference=pref,
                monadic_preference_builder=mpref_build,
                vg=geometries[pname],
            )

        world = TrajectoryWorld(map_name=map_name, scenario=scenario, geo=geometries, goals=goals)
        get_outcomes = partial(MetricEvaluation.evaluate, world=world)

        game = TrajectoryGame(
            world=world,
            game_players=players,
            ps=ps,
            get_outcomes=get_outcomes,
            game_vis=TrajGameVisualization(world=world, plot_limits="auto"),
        )
        return game

    @staticmethod
    def preprocess_game(game: TrajectoryGame) -> SolvingContext:
        def compute_actions_and_commands(game: TrajectoryGame) -> Mapping[PlayerName, FrozenSet[Trajectory]]:
            """Generate the trajectories (or load from pre-computed) for each player (i.e. get the available actions)"""
            print("\nGenerating Trajectories:")
            trajs_and_commands: Dict[PlayerName, FrozenSet[Trajectory]] = {}
            for player_name, game_player in game.game_players.items():
                if isinstance(game_player.actions_generator, Trajectory):
                    trajs_and_commands[player_name] = [game_player.actions_generator] #todo: Workaround because there is only one trajectory -> when there are multiple it will fail
                elif isinstance(game_player.actions_generator, TrajectoryGenerator):
                    states = game_player.state.support()
                    assert len(states) == 1, states
                    trajs_and_commands[player_name] = game_player.actions_generator.get_actions(state=next(iter(states)), return_commands=True)

            return trajs_and_commands

        def get_context(game: TrajectoryGame, actions: Mapping[PlayerName, FrozenSet[Trajectory]]) -> SolvingContext:
            # Similar to get_outcome_preferences_for_players, use SetPreference1 for Poss
            pref: Mapping[PlayerName, Preference[PlayerEvaluatedMetrics]] = {
                name: player.preference for name, player in game.game_players.items()
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
                "game_outcomes": game.get_outcomes,
                "outcome_pref": pref,
                "solver_params": solver_params,
            }

            return SolvingContext(**kwargs)

        trajs_and_commands = compute_actions_and_commands(game=game)

        def iterate_combinations(trajs_and_commands):
            # there is only one trajectory for 'P1'
            combinations = []
            current_mapping: Mapping[PlayerName, Trajectory] = {}
            P1 = PlayerName('P1')
            EGO = PlayerName('Ego')
            for traj, cmds in trajs_and_commands['Ego']:
                p1 = trajs_and_commands['P1'][0]
                ego = traj
                current_mapping: Mapping[PlayerName, Trajectory] #= {P1: p1, EGO: ego}
                current_mapping[P1]=p1
                current_mapping[EGO]=ego
                combinations.append(current_mapping)
            return combinations

        actions_list = iterate_combinations(trajs_and_commands)
        for joint_traj in actions_list:
        # for joint_traj in set(iterate_dict_combinations(trajs_and_commands)):
            game.get_outcomes(joint_traj)

        # for player, actions in available_trajectories.items():
        #     for traj in actions:
        #         game.get_outcomes(frozendict({player: traj}))

        return get_context(game=game, actions=available_trajectories)

    @time_function
    def on_episode_init(self, my_name: PlayerName):
        pass
        # self.my_name = my_name
        # self.game: TrajectoryGame = get_simple_traj_game_leon(self.config_str)
        # self.solving_context = preprocess_full_game(sgame=self.game, only_traj=False)
        # self.solution: Solution = Solution()
        # self.nash_eq: Mapping[str, SolvedTrajectoryGame] = self.solution.solve_game(context=self.solving_context)
        # folder = "example_game_playing_agent_leon/"
        # self.trajectory = self.nash_eq[my_name]['admissible'].actions[my_name]
        # if self.reporting:
        #     self.game.game_vis.init_plot_dict(values=self.nash_eq["weak"])
        #     report_single(game=self.game, nash_eq=self.nash_eq, folder=folder)

    def get_commands(self, sim_obs: SimObservations) -> U:
        # current_time = sim_obs.time
        # current_state = self.trajectory.at_interp(current_time)
        # return VehicleCommands(acc=current_state, ddelta: float)
        pass

    def on_get_extra(self) -> Optional[DrawableTrajectoryType]:
        pass
