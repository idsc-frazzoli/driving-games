from functools import partial
from time import perf_counter

from cytoolz import valmap
from frozendict import frozendict

from dg_commons import RJ, RP, U, X, Y
from dg_commons.utils_toolz import fd
from games.create_joint_game_graph import create_game_graph
from games.game_def import (
    Game,
    GamePlayer,
    SR,
)
from games.game_graph_to_nx import build_networkx_from_game_graph
from games.get_indiv_games import get_individual_games
from games.performance import PerformanceStatistics
from games.solve.solution import solve_game
from games.solve.solution_structures import (
    GameGraph,
    GamePlayerPreprocessed,
    GamePreprocessed,
    GameSolution,
    SolverParams,
)

__all__ = ["preprocess_game"]


def preprocess_game(
    game: Game[X, U, Y, RP, RJ, SR], solver_params: SolverParams, perf_stats: PerformanceStatistics
) -> GamePreprocessed[X, U, Y, RP, RJ, SR]:
    """
    1. Preprocesses the game computing the general game graph (MultiDiGraph used for visualisation)
    2. Computes the solutions for the single players
    3. If factorization is selected, computes the corresponding game factorization

    :param game:
    :param solver_params:
    :param perf_stats: Object used to collect performance statistics
    :return:
    """

    # get the individual game
    if solver_params.use_factorization:
        individual_games = get_individual_games(game)
        partial_preprocess_player = partial(preprocess_player, solver_params=solver_params, perf_stats=perf_stats)
        players_pre = valmap(partial_preprocess_player, individual_games)
    else:
        players_pre = fd({})

    gp = GamePreprocessed(
        game=game,
        players_pre=players_pre,
        game_graph_nx=None,
        solver_params=solver_params,
        perf_stats=perf_stats,
    )

    return gp


def preprocess_player(
    individual_game: Game[X, U, Y, RP, RJ, SR],
    solver_params: SolverParams,
    perf_stats: PerformanceStatistics,
) -> GamePlayerPreprocessed[X, U, Y, RP, RJ, SR]:
    """
    # Preprocess a single player by solving their individual games (i.e. optimal control problem)
    :param individual_game:
    :param solver_params:
    :return:
    """
    l = list(individual_game.players)
    assert len(l) == 1
    player_name = l[0]
    player: GamePlayer = individual_game.players[player_name]
    initials = frozenset(map(lambda x: frozendict({player_name: x}), player.initial.support()))
    assert len(initials) == 1

    tic = perf_counter()
    game_graph: GameGraph[X, U, Y, RP, RJ, SR]
    # create the actual game graph for the player
    game_graph = create_game_graph(
        game=individual_game,
        dt=solver_params.dt,
        initials=initials,
        players_pre=fd({}),
        fact_algo=solver_params.factorization_algorithm,
        max_depth=solver_params.max_depth,
    )
    tic2 = perf_counter()
    perf_stats.individual_game_graphs_nodes.append(len(game_graph.state2node))
    gs: GameSolution[X, U, Y, RP, RJ, SR]
    gs = solve_game(game=individual_game, solver_params=solver_params, gg=game_graph, initials=initials)
    toc = perf_counter()
    perf_stats.build_individual_game_graphs.append(tic2 - tic)
    perf_stats.solve_individual_game_graphs.append(toc - tic2)
    # create the NX game graph for the player
    graph_nx = build_networkx_from_game_graph(game_graph)
    return GamePlayerPreprocessed(graph_nx, game_graph, gs)


# def get_reachable_states(
#     initial: Poss[X],
#     personal_reward_structure: PersonalRewardStructure[X, U, RP],
#     dynamics: Dynamics[X, U, SR],
#     dt: D,
# ) -> MultiDiGraph:
#     """
#     Computes the states accessible for a player subject to their dynamics and their personal cost function.
#     :param initial:
#     :param personal_reward_structure:
#     :param dynamics:
#     :param dt:
#     :return:
#     """
#     check_poss(initial, object)
#     G = MultiDiGraph()
#
#     for node in initial.support():
#         i_final = personal_reward_structure.is_personal_final_state(node)
#         if i_final:
#             raise ZException(i_final=i_final)
#         G.add_node(node, is_final=False)
#
#     stack = list(initial.support())
#     # i: int = 0
#     expanded = set()
#     while stack:
#         # i += 1
#         s1 = stack.pop(0)
#         assert s1 in G.nodes
#         if s1 in expanded:
#             continue
#         # is_final =  player.personal_reward_structure.is_personal_final_state(s1)
#         # G.add_node(s1, is_final=is_final)
#         # # logger.info(s1=G.nodes[s1])
#
#         expanded.add(s1)
#         successors = dynamics.successors(s1, dt)
#         for u, p_s2 in successors.items():
#             check_poss(p_s2, object)
#             # for u, s2s in successors.items():
#             for s2 in p_s2.support():
#                 if s2 not in G.nodes:
#                     is_final2 = personal_reward_structure.is_personal_final_state(s2)
#                     G.add_node(s2, is_final=is_final2)
#                     if not is_final2:
#                         stack.append(s2)
#
#                 G.add_edge(s1, s2, u=u)
#     return G

###### this function is deprecated, it has been replaced by the function in game_graph_to_nx.py
# @time_function
# def build_networkx_game_graph(game: Game[X, U, Y, RP, RJ, SR], dt: D) -> MultiDiGraph:
#     """Gets the game graph, currently used only for visualisation.
#     Note that the real game is built in create_joint_game_tree.
#     If factorization is used this game graph will differ from the one that is actually solved."""
#     players = game.players
#     init_states: Mapping[PlayerName, X] = valmap(lambda x: x.initial.support(), players)
#
#     G = MultiDiGraph()
#     stack: List[JointState] = []
#     # root of the tree
#     for S in iterate_dict_combinations(init_states):
#         G.add_node(
#             S,
#             is_joint_final_for="",
#             is_pers_final="",
#             is_initial=True,
#             is_terminal=False,
#             generation=0,
#             in_game="-".join(S.keys()),
#         )
#         stack.append(S)
#     logger.info(stack=stack)
#     # all the rest of the tree
#     i = 0
#     S: JointState
#     ps = game.ps
#     while stack:
#         if i % 1000 == 0:
#             logger.info("Iteration", i=i, stack=len(stack), created=len(G.nodes))
#         i += 1
#         S = stack.pop()
#         assert S in G.nodes
#
#         players_alive = filter(
#             lambda x: x not in G.nodes[S]["is_joint_final_for"] and x not in G.nodes[S]["is_pers_final"], S
#         )
#         successors: Dict[PlayerName : Mapping[U, Poss[X]]] = {}
#         for p in players_alive:
#             p_state = S[p]
#             p_succs = players[p].dynamics.successors(p_state, dt)
#             successors[p] = p_succs
#
#         generation = G.nodes[S]["generation"]
#         for players_n_actions in iterate_dict_combinations(successors):
#             poss_next = [successors[p][action].support() for p, action in players_n_actions.items()]
#             players_poss_next = dict(zip(players_n_actions, poss_next))
#             for S2 in iterate_dict_combinations(players_poss_next):
#                 if S2 not in G.nodes:
#                     personal_ending = {
#                         p for p in S2 if players[p].personal_reward_structure.is_personal_final_state(S2[p])
#                     }
#                     transitions = {p: DgSampledSequence[X](timestamps=(D(0), dt), values=(S[p], S2[p])) for p in S2}
#                     jointly_ending = game.joint_reward.is_joint_final_transition(transitions)
#                     ending_players = jointly_ending | personal_ending
#                     still_alive: bool = any(p not in ending_players for p in S2)
#                     G.add_node(
#                         S2,
#                         is_joint_final_for="-".join(jointly_ending),
#                         is_pers_final="-".join(personal_ending),
#                         is_initial=False,
#                         is_terminal=not still_alive,
#                         generation=generation + 1,
#                         in_game="-".join(S2.keys()),
#                     )
#                     # if anyone is still alive add to stack for further expansion
#                     if still_alive:
#                         if S2 not in stack:
#                             stack.append(S2)
#                 G.add_edge(S, S2, action=players_n_actions)
#                 G.nodes[S2]["generation"] = min(G.nodes[S2]["generation"], generation + 1)
#     logger.info("Game nodes", created=len(G.nodes))
#     return G


# def get_player_graph(player: GamePlayer[X, U, Y, RP, RJ, SR], dt: D) -> MultiDiGraph:
#     return get_reachable_states(player.initial, player.personal_reward_structure, player.dynamics, dt=dt)
