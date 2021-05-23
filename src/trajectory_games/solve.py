from copy import deepcopy
from itertools import product
from random import choice
from time import perf_counter
from typing import Mapping, Dict, FrozenSet, Set, Tuple, Optional, List
from frozendict import frozendict

from games.utils import iterate_dict_combinations
from preferences import ComparisonOutcome, SECOND_PREFERRED, INDIFFERENT, INCOMPARABLE, FIRST_PREFERRED, Preference

from games import PlayerName
from .game_def import SolvingContext, EXP_ACCOMP
from .trajectory_game import JointPureTraj, SolvedTrajectoryGameNode, SolvedTrajectoryGame, \
    SolvedLeaderFollowerGame, LeaderFollowerGameSolvingContext, PrefsTup, LeaderFollowerGameNode
from .paths import Trajectory
from .metrics_def import TrajGameOutcome, PlayerOutcome, Metric, EvaluatedMetric

JointTrajSet = Mapping[PlayerName, FrozenSet[Trajectory]]
EqOutcome = Tuple[Optional[JointPureTraj], Optional[TrajGameOutcome], bool, bool, bool, bool]
NotEq: EqOutcome = None, None, False, False, False, False


def get_solved_game_node(act: JointPureTraj, out: TrajGameOutcome) -> SolvedTrajectoryGameNode:
    return SolvedTrajectoryGameNode(actions=act, outcomes=out)


def callback_eq(tuple_out: EqOutcome, eq: Dict[str, SolvedTrajectoryGame]):
    joint_act, outcome, strong, incomp, indiff, weak = tuple_out

    solved_node: SolvedTrajectoryGameNode = \
        get_solved_game_node(act=joint_act, out=outcome) if weak else None
    eq["indiff"].add(solved_node) if indiff else None
    eq["incomp"].add(solved_node) if incomp else None
    eq["weak"].add(solved_node) if weak else None
    eq["strong"].add(solved_node) if strong else None


def init_eq_dict() -> Dict[str, SolvedTrajectoryGame]:
    ret: Dict[str, SolvedTrajectoryGame] = {
        "indiff": set(), "incomp": set(),
        "weak": set(), "strong": set(),
    }
    return ret


def check_dominated(joint_actions: JointPureTraj,
                    done: Mapping[PlayerName, Set[JointPureTraj]]) -> bool:
    for dominated in done.values():
        if joint_actions in dominated:
            return True
    return False


def get_best_responses(joint_actions: JointPureTraj, context: SolvingContext,
                       player: PlayerName, done_p: Set[JointPureTraj],
                       player_pref: Preference = None) \
        -> Tuple[Set[ComparisonOutcome], Set[Trajectory]]:
    """
    Calculates the best responses for the current player
    Returns best responses and comparison outcomes
    """
    actions = context.player_actions
    players = joint_actions.keys()
    if player_pref is None:
        player_pref = context.outcome_pref[player]

    # All alternate actions (from available set) for the current player
    def get_action_options(joint_act: JointPureTraj, p_actions: Set[Trajectory]) -> JointTrajSet:
        """Returns all possible actions for the player, with other player actions frozen
        Current player action is not included"""

        def get_actions(name: PlayerName) -> FrozenSet[Trajectory]:
            if name == player:
                return frozenset(p_actions)
            return frozenset({joint_act[name]})

        action_options: JointTrajSet = {_: get_actions(_) for _ in players}
        return action_options

    all_actions: Set[Trajectory] = set(actions[player])
    all_actions.remove(joint_actions[player])

    # Save antichain of actions
    best: Set[Trajectory] = {joint_actions[player]}
    results: Set[ComparisonOutcome] = set()
    check = False
    action_alt: JointTrajSet = get_action_options(joint_act=joint_actions,
                                                  p_actions=all_actions)
    for joint_act_alt in iterate_dict_combinations(action_alt):
        if joint_act_alt in done_p:
            continue
        check = True
        alt_action: Trajectory = joint_act_alt[player]
        alt_outcome: PlayerOutcome = context.game_outcomes(joint_act_alt)[player]
        joint_best_all = get_action_options(joint_act=joint_actions, p_actions=best)
        results_alt: Set[ComparisonOutcome] = set()

        for joint_best in set(iterate_dict_combinations(joint_best_all)):
            best_outcome: PlayerOutcome = context.game_outcomes(joint_best)[player]
            comp_outcome: ComparisonOutcome = player_pref.compare(best_outcome, alt_outcome)
            results_alt.add(comp_outcome)
            if joint_best == joint_actions:
                results.add(comp_outcome)

            # If one of best is preferred, alternate is not a best response
            if comp_outcome == FIRST_PREFERRED:
                done_p.add(joint_act_alt)
                break

            # If second option is preferred, current best action is not a best response
            if comp_outcome == SECOND_PREFERRED:
                best.remove(joint_best[player])
                done_p.add(joint_best)

        if FIRST_PREFERRED not in results_alt:
            best.add(alt_action)

    if not check:
        results.add(FIRST_PREFERRED)
    return results, best


def equilibrium_check(joint_actions: JointPureTraj, context: SolvingContext,
                      done: Dict[PlayerName, Set[JointPureTraj]]) -> EqOutcome:
    """
    For each player, check if current action is best response
    Classify into types of nash eq. based on the outputs
    """
    # TODO[SIR]: Extend to mixed outcomes and strategies

    if isinstance(joint_actions, dict):
        joint_actions = frozendict(joint_actions)
    if check_dominated(joint_actions=joint_actions, done=done):
        return NotEq

    # Compute best responses for each player, check if joint_actions is dominated
    results: Set[ComparisonOutcome] = set()
    for player in joint_actions.keys():
        results_player, _ = get_best_responses(joint_actions=joint_actions, context=context,
                                               player=player, done_p=done[player])
        # If second option is preferred for any player, current point is not a nash eq.
        if SECOND_PREFERRED in results_player:
            return NotEq
        results |= results_player

    outcome: TrajGameOutcome = context.game_outcomes(joint_actions)
    strong = results == {FIRST_PREFERRED}
    indiff = INDIFFERENT in results
    incomp = INCOMPARABLE in results
    weak = True  # All equilibria are atleast weak
    return joint_actions, outcome, strong, incomp, indiff, weak


def calculate_expectation(outcomes: List[PlayerOutcome]) -> PlayerOutcome:
    n_out = len(outcomes)
    if n_out == 0:
        raise AssertionError("Received empty input for calculate_expectation!")
    if n_out == 1:
        return frozendict({m: em for m, em in outcomes[0].items()})

    def init_eval_metric(evalm: EvaluatedMetric) -> EvaluatedMetric:
        return EvaluatedMetric(title=evalm.title, description=evalm.description, total=0.0,
                               incremental=None, cumulative=None)

    total: Dict[Metric, EvaluatedMetric] = {m: init_eval_metric(evalm=em) for m, em in outcomes[0].items()}
    for out in outcomes:
        for m, em in out.items():
            total[m].total += em.total
    for m in outcomes[0]:
        total[m].total /= n_out
    return frozendict(total)


def calculate_join(outcomes: Mapping[Trajectory, PlayerOutcome]) -> PlayerOutcome:
    # TODO[SIR]: Implement this after testing expectation
    pass


def get_security_strategies(players: Tuple[PlayerName, PlayerName], context: SolvingContext) \
        -> Mapping[Trajectory, Tuple[SolvedTrajectoryGame, PlayerOutcome]]:
    """
    Calculates the security strategies of the leader
    """
    lead, foll = players
    all_actions: Dict[Trajectory, PlayerOutcome] = {}
    all_games: Dict[Trajectory, SolvedTrajectoryGame] = {}
    use_best_resp: bool = context.solver_params.use_best_response
    lead_actions = context.player_actions[lead]
    foll_actions = {_ for _ in context.player_actions[foll]}
    foll_act_1 = next(iter(foll_actions))
    for l_act in lead_actions:
        best_resp: Set[Trajectory]
        if use_best_resp:
            joint_act: Dict[PlayerName, Trajectory] = {lead: l_act, foll: foll_act_1}
            _, best_resp = get_best_responses(joint_actions=joint_act, context=context,
                                              player=foll, done_p=set())
        else:
            best_resp = foll_actions

        outcomes: List[PlayerOutcome] = []
        game_nodes: SolvedTrajectoryGame = set()
        for f_act in best_resp:
            joint_act = {lead: l_act, foll: f_act}
            out = frozendict(context.game_outcomes(joint_act))
            outcomes.append(out[lead])
            game_nodes.add(SolvedTrajectoryGameNode(actions=frozendict(joint_act), outcomes=out))
        if context.solver_params.antichain_comparison == EXP_ACCOMP:
            all_actions[l_act] = calculate_expectation(outcomes=outcomes)
        else:
            raise NotImplementedError("Join antichain comparison not yet implemented")
        all_games[l_act] = game_nodes

    for act in set(all_actions.keys()):
        if act not in all_actions:
            continue
        for act_alt in set(all_actions.keys()):
            comp = context.outcome_pref[lead].compare(all_actions[act], all_actions[act_alt])
            if comp == SECOND_PREFERRED:
                all_actions.pop(act)
                break
            elif comp == FIRST_PREFERRED:
                all_actions.pop(act_alt)
    solved_games: Dict[Trajectory, Tuple[SolvedTrajectoryGame, PlayerOutcome]] = \
        {node: (all_games[node], frozendict(all_actions[node])) for node in all_actions.keys()}
    return solved_games


class Solution:
    # Cache can be reused between levels
    dominated: Dict[PlayerName, Set[JointPureTraj]] = None

    def solve_game(self, context: SolvingContext, cache_dom: bool = False) \
            -> Mapping[str, SolvedTrajectoryGame]:
        eq_dict = init_eq_dict()
        dom_prev = deepcopy(self.dominated) if not cache_dom else {}
        tic = perf_counter()
        # For each possible action combination, check if it is a nash eq
        if self.dominated is None:
            self.dominated = {_: set() for _ in context.player_actions.keys()}
        for joint_act in set(iterate_dict_combinations(context.player_actions)):
            out = equilibrium_check(joint_actions=joint_act, context=context,
                                    done=self.dominated)
            callback_eq(tuple_out=out, eq=eq_dict)
        toc = perf_counter() - tic
        print(f"Nash equilibrium computation time = {toc:.2f} s")
        if not cache_dom:
            self.dominated = dom_prev
        return eq_dict

    def reset(self):
        self.dominated: Dict[PlayerName, Set[JointPureTraj]] = None


def solve_leader_follower(context: LeaderFollowerGameSolvingContext) \
        -> SolvedLeaderFollowerGame:
    lf = context.lf

    if not context.solver_params.use_best_response:
        raise NotImplementedError("Leader follower assumes follower best response!")

    tic1 = perf_counter()
    # Collect all possible actions for players
    lead_actions = set(context.player_actions[lf.leader])
    foll_actions = set(context.player_actions[lf.follower])
    foll_act_1 = next(iter(foll_actions))

    tic = perf_counter()
    # Calculate best responses and corresponding leader outcomes
    # -> for every leader action and follower preference
    br_pref: Dict[Trajectory, Dict[Preference, Set[Trajectory]]] = {}
    out_l: Dict[Trajectory, Dict[Preference, List[PlayerOutcome]]] = {}
    for l_act in lead_actions:
        br_pref[l_act], out_l[l_act] = {}, {}
        joint_act: Dict[PlayerName, Trajectory] = {lf.leader: l_act, lf.follower: foll_act_1}
        for p_f in lf.prefs_follower:
            _, br = get_best_responses(joint_actions=joint_act, context=context,
                                       player=lf.follower, done_p=set(),
                                       player_pref=p_f)

            outcomes_l: List[PlayerOutcome] = []
            for act in iterate_dict_combinations({lf.leader: {l_act}, lf.follower: br}):
                out = frozendict(context.game_outcomes(act))
                outcomes_l.append(out[lf.leader])
            br_pref[l_act][p_f] = br
            out_l[l_act][p_f] = outcomes_l

    toc = perf_counter() - tic
    print(f"Best response time = {toc:.2f} s")

    # Calculate aggregated leader outcomes
    # -> for each leader action and prefs of both players
    tic = perf_counter()
    agg_out_l: Dict[Trajectory, Dict[PrefsTup, PlayerOutcome]] = {}
    for l_act in lead_actions:
        agg_out_l_act = {}
        for p_l, p_f in product(lf.prefs_leader, lf.prefs_follower):
            if context.solver_params.antichain_comparison == EXP_ACCOMP:
                agg_out_l_act[(p_l, p_f)] = calculate_expectation(outcomes=out_l[l_act][p_f])
            else:
                raise NotImplementedError("Join antichain comparison not yet implemented")
        agg_out_l[l_act] = agg_out_l_act
    toc = perf_counter() - tic
    print(f"Agg outcomes time = {toc:.2f} s")

    # Calculate best actions of leader and all possible best actions
    # For every pref combination, compare all agg lead outcomes and select non-dominated ones
    tic = perf_counter()
    best_actions: Dict[PrefsTup, Set[Trajectory]] = {}
    all_actions: Set[Trajectory] = set()
    for p_l, p_f in product(lf.prefs_leader, lf.prefs_follower):
        ba_pref: Set[Trajectory] = set(lead_actions)
        for act_1 in frozenset(ba_pref):
            if act_1 not in ba_pref:
                continue
            for act_2 in frozenset(ba_pref):
                if act_1 == act_2:
                    continue
                comp = p_l.compare(agg_out_l[act_1][(p_l, p_f)], agg_out_l[act_2][(p_l, p_f)])
                if comp == SECOND_PREFERRED:
                    ba_pref.remove(act_1)
                    break
                elif comp == FIRST_PREFERRED:
                    ba_pref.remove(act_2)
        best_actions[(p_l, p_f)] = ba_pref
        all_actions |= ba_pref

    toc = perf_counter() - tic
    print(f"Best leader actions time = {toc:.2f} s")

    # Calculate final outcomes and post-process for data structure
    game_nodes: Dict[Trajectory, Dict[PrefsTup, LeaderFollowerGameNode]] = {}
    for l_act in all_actions:
        pref_nodes: Dict[PrefsTup, LeaderFollowerGameNode] = {}
        for p_l, p_f in product(lf.prefs_leader, lf.prefs_follower):
            solved_game: SolvedTrajectoryGame = set()
            for act in iterate_dict_combinations({lf.leader: {l_act}, lf.follower: br_pref[l_act][p_f]}):
                out = context.game_outcomes(act)
                solved_game.add(SolvedTrajectoryGameNode(actions=act, outcomes=out))
            pref_nodes[(p_l, p_f)] = LeaderFollowerGameNode(nodes=solved_game,
                                                            agg_lead_outcome=agg_out_l[l_act][(p_l, p_f)])
        game_nodes[l_act] = pref_nodes

    toc = perf_counter() - tic1
    print(f"Game solving time = {toc:.2f} s")

    return SolvedLeaderFollowerGame(lf=lf, games=game_nodes, best_leader_actions=best_actions)


def solve_static_game(context: SolvingContext) \
        -> Mapping[str, SolvedTrajectoryGame]:
    sol = Solution()
    eq_dict = sol.solve_game(context=context)
    static_eq: Dict[str, SolvedTrajectoryGame] = {}

    for eq_type, node_set in eq_dict.items():
        static_set = {SolvedTrajectoryGameNode(actions=node.actions, outcomes=node.outcomes)
                      for node in node_set}
        static_eq[eq_type] = static_set

    return static_eq


def iterative_best_response(context: SolvingContext, n_runs: int) \
        -> Mapping[str, SolvedTrajectoryGame]:
    eq_dict = init_eq_dict()
    INIT_BEST = True

    # Solve single player game for each player to get initial guess
    all_actions: Mapping[PlayerName, FrozenSet[Trajectory]] = context.player_actions
    init_guess: Dict[PlayerName, Set[Trajectory]] = {}
    done_p: Set[JointPureTraj] = set()
    tic = perf_counter()
    if INIT_BEST:
        print("Initialising strategies using best personal strategies.\nBest Actions:")
        for player, actions in all_actions.items():
            done_p.clear()
            joint_actions = frozendict({player: next(iter(actions))})
            _, best_actions = get_best_responses(joint_actions=joint_actions, context=context,
                                                 player=player, done_p=done_p)
            print(f"\tPlayer: {player} = {len(best_actions)}")
            init_guess[player] = best_actions
        toc = perf_counter() - tic
        print(f"Individual best computation time = {toc:.2f} s")
    else:
        print("Initialising strategies at random")
        init_guess = {player: actions for player, actions in all_actions.items()}

    done: Dict[PlayerName, Set[JointPureTraj]] = \
        {_: set() for _ in context.player_actions.keys()}
    for i in range(n_runs):
        players_rem: Set[PlayerName] = set(init_guess.keys())
        joint_best: Dict[PlayerName, Trajectory] = {}
        for player, actions in init_guess.items():
            joint_best[player] = choice(list(actions))

        # Select a player at random and change action to one from best response set
        # Continue till joint action is part of best for all players
        while len(players_rem) > 0:
            player = choice(list(players_rem))
            players_rem.remove(player)
            _, best_actions = get_best_responses(joint_actions=joint_best, context=context,
                                                 player=player, done_p=done[player])
            if joint_best[player] not in best_actions:
                players_rem = set(init_guess.keys())
                players_rem.remove(player)
            joint_best[player] = choice(list(best_actions))

        out = equilibrium_check(joint_actions=joint_best, context=context, done=done)
        callback_eq(tuple_out=out, eq=eq_dict)

    toc = perf_counter() - tic
    print(f"Best response equilibrium computation time = {toc:.2f} s")

    return eq_dict
