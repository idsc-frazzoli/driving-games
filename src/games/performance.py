from dataclasses import dataclass
from typing import Dict
from games.game_def import PlayerName, JointState, Game
from games.solve.solution_structures import AdmissibleStrategies, StrategyForMultipleNash, SolverParams
from copy import deepcopy


@dataclass
class PerformanceInfo:
    """
    Base class for containing infos of the game solving process
    """
    total_time: float
    """Total time in seconds spend for the episode"""


@dataclass
class GetFactorizationPI(PerformanceInfo):
    """
    Contains performance information of the factorization process
    """
    find_dependencies_times: Dict[JointState, float]
    """For each joint state collects the number of seconds to find the dependencies between players"""

    total_time_find_dependencies: float
    """The total time spend to find the dependencies"""

    total_time_collision_check: float
    """ The total time to check collision """

    def __repr__(self):
        return (f"GetFactorizationPI(total_time={self.total_time}, "
                f"total_time_find_dependencies={self.total_time_find_dependencies})"
                )

    __str__ = __repr__


@dataclass
class CreateGameGraphPI(PerformanceInfo):
    """
    Contains performance information of the game graph creation
    """
    check_collision_total_time: float
    get_shared_resources_total_time: float


@dataclass
class GetFutResourcesPI(PerformanceInfo):
    """
    Contains performance information of the collection of the future resources
    """
    pass


@dataclass
class SolveGamePI(PerformanceInfo):
    """
    Contains performance information of the solving process
    """
    get_resources_pi: GetFutResourcesPI
    """Contains the performance information of collecting the future resources of a game"""


@dataclass
class PreprocessPlayerPI(PerformanceInfo):
    """
    Contains the performance info of the preprocessing step of the players
    """
    get_fact_pi: GetFactorizationPI
    """The performance info of the factorization step"""

    create_game_tree_pre_pi: Dict[PlayerName, CreateGameGraphPI]
    """The performance information of creating the game tree for each player"""

    solve_game_pre_pi: Dict[PlayerName, SolveGamePI]
    """The performance information of solving the single player game for each player"""

    @property
    def total_time_get_fut_resources(self) -> float:
        """
        The total time spend for creating all the single player game trees
        """
        return sum([sg_pi.get_resources_pi.total_time for sg_pi in self.solve_game_pre_pi.values()])

    @property
    def total_time_solve_game(self) -> float:
        """
        The total time spend for solving all the single player games
        """
        return sum([sg_pi.total_time for sg_pi in self.solve_game_pre_pi.values()])

    def __repr__(self):
        return (f"PreprocessPlayerPI(total_time={self.total_time}, "
                f"total_time_get_fut_resources={self.total_time_get_fut_resources}, "
                f"total_time_solve_game={self.total_time_solve_game})"
                )

    __str__ = __repr__


@dataclass
class GamePerformance:
    """
    Contains general information about the performance of solving the game
    """
    nb_players: int
    """How many players"""

    size_action_set: Dict[PlayerName, int]
    """The size of the action set for each player"""

    strat_mult_nash: StrategyForMultipleNash
    """The strategy when finding multiple Nash equilibra"""

    adm_strat: AdmissibleStrategies
    """What strategies of players are considered while solving the game"""

    use_fact: bool
    """Indicates if factorization was used"""

    pre_pro_player_pi: PreprocessPlayerPI
    """Contains the performance information for the preprocessing of the players"""

    create_gt_pi: CreateGameGraphPI
    """Contains the performance information for creating the joint game tree"""

    solve_game_pi: SolveGamePI
    """Contains the performance information for solving the joint game"""


def get_initialized_game_performance(
        game: Game, solver_params: SolverParams
) -> GamePerformance:

    create_gg_pi = CreateGameGraphPI(
        total_time=0,
        check_collision_total_time=0,
        get_shared_resources_total_time=0
    )

    get_res_pi = GetFutResourcesPI(
        total_time=0
    )

    solve_game_pi = SolveGamePI(
        total_time=0,
        get_resources_pi=deepcopy(get_res_pi),
    )

    get_fact_pi = GetFactorizationPI(
        total_time=0,
        find_dependencies_times={},
        total_time_find_dependencies=0,
        total_time_collision_check=0
    )

    size_action_set: Dict[PlayerName, int] = {}
    create_game_tree_pre_pi: Dict[PlayerName, CreateGameGraphPI] = {}
    solve_game_pre_pi: Dict[PlayerName, SolveGamePI] = {}
    for pn, gp in game.players.items():
        size_action_set[pn] = len(gp.dynamics.all_actions())
        create_game_tree_pre_pi[pn] = deepcopy(create_gg_pi)
        solve_game_pre_pi[pn] = deepcopy(solve_game_pi)

    pre_pro_pl_pi = PreprocessPlayerPI(
        total_time=0,
        get_fact_pi=deepcopy(get_fact_pi),
        create_game_tree_pre_pi=create_game_tree_pre_pi,
        solve_game_pre_pi=solve_game_pre_pi,
    )

    nb_players = len(game.players)
    strat_mult_nash: StrategyForMultipleNash = solver_params.strategy_multiple_nash
    adm_strat: AdmissibleStrategies = solver_params.admissible_strategies
    use_fact = solver_params.use_factorization

    game_perf = GamePerformance(
        nb_players=nb_players,
        size_action_set=size_action_set,
        strat_mult_nash=strat_mult_nash,
        adm_strat=adm_strat,
        use_fact=use_fact,
        pre_pro_player_pi=pre_pro_pl_pi,
        create_gt_pi=deepcopy(create_gg_pi),
        solve_game_pi=deepcopy(solve_game_pi)
    )

    return game_perf

