from decimal import Decimal as D
from itertools import product
from typing import (
    List,
    Optional,
)

from frozendict import frozendict
from networkx import MultiDiGraph
from toolz import  valmap

from possibilities import check_poss
from games import logger
from games.game_def import (
    Dynamics,
    Game,
    GamePlayer,
    JointState,
    PersonalRewardStructure,
    PlayerName,
    RJ,
    RP,
    SR,
    U,
    X,
    Y,
)
from games.get_indiv_games import get_individual_games
from games.access import preprocess_player, compute_graph_layout
from games.solve.solution_structures import (
    GameFactorization,
    GamePreprocessed,
)

from factorization.structures import FactorizationSolverParams


def preprocess_duckie_game(
    game: Game[X, U, Y, RP, RJ, SR],
    solver_params: FactorizationSolverParams,
) -> GamePreprocessed[X, U, Y, RP, RJ, SR]:
    """
    1. Preprocesses the game computing the general game graph for only two players (MultiDiGraph used for visualisation)
    2. Computes the solutions for the single players
    3. If factorization is selected, computes the corresponding game factorization

    :param game:
    :param solver_params:
    :return:
    """
    game_factorization: Optional[GameFactorization[X]] = None

    game_graph = get_duckie_game_graph(game, dt=solver_params.dt)
    compute_graph_layout(game_graph, iterations=1)
    individual_games = get_individual_games(game)
    players_pre = valmap(
        lambda individual_game: preprocess_player(
            solver_params=solver_params, individual_game=individual_game
        ),
        individual_games,
    )
    if solver_params.use_factorization:
        game_factorization = solver_params.get_factorization(game, players_pre)

    gp = GamePreprocessed(
        game=game,
        players_pre=players_pre,
        game_graph=game_graph,
        solver_params=solver_params,
        game_factorization=game_factorization,
    )

    return gp


def get_duckie_game_graph(game: Game[X, U, Y, RP, RJ, SR], dt: D) -> MultiDiGraph:
    """ Returns the game for the first two players"""
    players = game.players
    # assert len(players) == 2
    p1, p2 = list(players)[0:2]
    P1 = players[p1]
    P2 = players[p2]
    # G1 = get_player_graph(players[p1])
    # G2 = get_player_graph(players[p2])

    G = MultiDiGraph()
    stack: List[JointState] = []
    # root of the tree
    for n1, n2 in product(P1.initial.support(), P2.initial.support()):
        S = frozendict({p1: n1, p2: n2})
        G.add_node(
            S,
            is_final2=False,
            is_final1=False,
            is_joint_final=False,
            is_initial=True,
            generation=0,
            in_game="AB",
        )
        stack.append(S)
    logger.info(stack=stack)
    # all the rest of the tree
    i = 0
    S: JointState
    ps = game.ps
    while stack:
        if i % 1000 == 0:
            logger.info("iteration", i=i, stack=len(stack), created=len(G.nodes))
        i += 1
        S = stack.pop()
        assert S in G.nodes

        n1, n2 = S[p1], S[p2]

        if n1 is None or G.nodes[S]["is_final1"]:
            succ1 = {None: ps.unit(None)}
        else:
            succ1 = P1.dynamics.successors(n1, dt)

        if n2 is None or G.nodes[S]["is_final2"]:
            succ2 = {None: ps.unit(None)}
        else:
            succ2 = P2.dynamics.successors(n2, dt)

        generation = G.nodes[S]["generation"]

        for (u1, s1s), (u2, s2s) in product(succ1.items(), succ2.items()):
            check_poss(s1s, object)
            check_poss(s2s, object)
            for s1, s2 in product(s1s.support(), s2s.support()):
                if (s1, s2) == (None, None):
                    continue
                S2 = frozendict({p1: s1, p2: s2})
                if S2 not in G.nodes:
                    is_final1 = P1.personal_reward_structure.is_personal_final_state(s1) if s1 else True
                    is_final2 = P2.personal_reward_structure.is_personal_final_state(s2) if s2 else True

                    in_game = "AB" if (s1 and s2) else ("A" if s1 else "B")
                    if s1 and s2:
                        is_joint_final = len(game.joint_reward.is_joint_final_state({p1: s1, p2: s2})) > 0
                    else:
                        is_joint_final = False
                    G.add_node(
                        S2,
                        is_final2=is_final2,
                        is_final1=is_final1,
                        is_joint_final=is_joint_final,
                        is_initial=False,
                        generation=generation + 1,
                        in_game=in_game,
                    )
                    if not is_joint_final:
                        if S2 not in stack:
                            stack.append(S2)
                G.add_edge(S, S2, action=frozendict({p1: u1, p2: u2}))
                G.nodes[S2]["generation"] = min(G.nodes[S2]["generation"], generation + 1)
    return G