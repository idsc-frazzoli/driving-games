from dataclasses import dataclass
from typing import Dict, TypeVar, Optional
from games.game_def import PlayerName, Game, Dynamics
from games.solve.solution_structures import AdmissibleStrategies, StrategyForMultipleNash, SolverParams
from copy import deepcopy
from toolz import valmap, valfilter
from numbers import Number

__all__ = [
    "GamePerformance",
    "get_initialized_game_performance",
    "PerformanceInfo",
    "GetFactorizationPI",
    "GetFutResourcesPI",
    "CreateGameGraphPI",
    "SolveGamePI",
    "PreprocessPlayerPI",
]


G = TypeVar("G")
""" Generic variable for a player's geometrie."""


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

    total_time_find_dependencies: float
    """The total time spend to find the dependencies"""

    total_time_find_dependencies_create_game_graph: float
    """The total time to create the game graphs during factorization for more than 2 players"""

    total_time_find_dependencies_solve_game: float
    """The total time to solve the game graphs during factorization for more than 2 players"""

    total_time_collision_check: float
    """ The total time to check collision """


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
                f"get_fact_pi={self.get_fact_pi}, "
                f"create_game_tree_pre_pi={self.create_game_tree_pre_pi}, "
                f"solve_game_pre_pi={self.solve_game_pre_pi}, "
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

    player_dynamics: Dict[PlayerName, Dynamics]
    """The dynamics of the players (only attributes which are numbers are collected)"""

    player_geometries: Dict[PlayerName, Optional[G]]
    """The geometries of a player if any"""

    use_fact: bool
    """Indicates if factorization was used"""

    beta: float
    """ Indicates the standard deviation of the gaussian kernel during the filtering of the mixed strategy"""

    pre_pro_player_pi: PreprocessPlayerPI
    """Contains the performance information for the preprocessing of the players"""

    create_gt_pi: CreateGameGraphPI
    """Contains the performance information for creating the joint game tree"""

    solve_game_pi: SolveGamePI
    """Contains the performance information for solving the joint game"""


def get_initialized_game_performance(
        game: Game, solver_params: SolverParams
) -> GamePerformance:
    """
    Returns an initialized game performance class.
    All times are set to zero and the solver specifications are filled in.

    :param game: The game that gets solved
    :param solver_params: The solver parameters
    :return: The initialized game performance class
    """

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
        total_time_find_dependencies=0,
        total_time_find_dependencies_create_game_graph=0,
        total_time_find_dependencies_solve_game=0,
        total_time_collision_check=0,
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
    player_dynamics = {pn: gp.dynamics.__dict__ for pn, gp in game.players.items()}
    player_dynamics_numbers_only = valmap(  # only display the objects that are "Numbers"
        lambda _: valfilter(lambda _val: isinstance(_val, Number), _),
        player_dynamics
    )

    def _get_geometries(_):
        if hasattr(_.dynamics, "vg"):  # check if the class has vehicle geometries
            return _.dynamics.vg.__dict__
        else:
            return None

    player_geometries = valmap(_get_geometries, game.players)

    strat_mult_nash: StrategyForMultipleNash = solver_params.strategy_multiple_nash
    adm_strat: AdmissibleStrategies = solver_params.admissible_strategies
    use_fact = solver_params.use_factorization

    if hasattr(solver_params, "beta"):  # todo make beta standard for all classes
        beta = solver_params.beta
    else:
        beta = 0

    game_perf = GamePerformance(
        nb_players=nb_players,
        size_action_set=size_action_set,
        strat_mult_nash=strat_mult_nash,
        adm_strat=adm_strat,
        player_dynamics=player_dynamics_numbers_only,
        player_geometries=player_geometries,
        use_fact=use_fact,
        pre_pro_player_pi=pre_pro_pl_pi,
        create_gt_pi=deepcopy(create_gg_pi),
        solve_game_pi=deepcopy(solve_game_pi),
        beta=beta
    )

    return game_perf
