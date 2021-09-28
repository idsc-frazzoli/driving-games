import math
from copy import deepcopy
from random import choice
from time import perf_counter
from typing import Mapping, Dict, Set, List, Callable, Tuple, Optional

from duckietown_world import SE2Transform
from frozendict import frozendict
from shapely.geometry import Polygon

from dg_commons.planning.lanes import DgLanelet
from games.utils import iterate_dict_combinations
from possibilities import Poss, PossibilityMonad
from preferences import ComparisonOutcome, SECOND_PREFERRED, FIRST_PREFERRED, Preference

from games import PlayerName
from .game_def import EXP_ACCOMP, JOIN_ACCOMP, SolvingContext
from .structures import VehicleState, VehicleGeometry
from .trajectory_game import (
    SolvedTrajectoryGameNode,
    SolvedTrajectoryGame,
    SolvedLeaderFollowerGame,
    LeaderFollowerGameSolvingContext,
    LeaderFollowerGameNode,
    LeaderFollowerGameStage,
    LeaderFollowerGame,
    preprocess_full_game,
    SolvedRecursiveLeaderFollowerGame,
)
from dg_commons.sequence import Timestamp, DgSampledSequence
from .paths import Trajectory
from .metrics_def import PlayerOutcome, Metric, EvaluatedMetric
from .metrics import Clearance
from .solve import get_best_responses


def init_eval_metric(evalm: EvaluatedMetric, total: float = 0.0) -> EvaluatedMetric:
    return EvaluatedMetric(
        title=evalm.title, description=evalm.description, total=total, incremental=None, cumulative=None
    )


def calculate_expectation(outcomes: List[PlayerOutcome]) -> PlayerOutcome:
    n_out = len(outcomes)
    if n_out == 0:
        raise AssertionError("Received empty input for calculate_expectation!")
    if n_out == 1:
        return frozendict({m: em for m, em in outcomes[0].items()})

    total: Dict[Metric, EvaluatedMetric] = {m: init_eval_metric(evalm=em) for m, em in outcomes[0].items()}
    for out in outcomes:
        for m, em in out.items():
            total[m].total += em.total
    for m in outcomes[0]:
        total[m].total /= n_out
    return frozendict(total)


def calculate_join(outcomes: List[PlayerOutcome], pref: Preference) -> PlayerOutcome:
    ac_worst = set(outcomes)
    for out1 in frozenset(ac_worst):
        if out1 not in ac_worst:
            continue
        for out2 in frozenset(ac_worst):
            comp: ComparisonOutcome = pref.compare(out1, out2)
            if comp == FIRST_PREFERRED:
                ac_worst.remove(out1)
                break
            elif comp == SECOND_PREFERRED:
                ac_worst.remove(out2)

    join: Dict[Metric, EvaluatedMetric] = {
        m: init_eval_metric(evalm=em, total=-math.inf) for m, em in outcomes[0].items()
    }
    # TODO[SIR]: This can be improved to calc more accurate joins
    for out in ac_worst:
        for m, em in out.items():
            if join[m].total < em.total:
                join[m].total = em.total
    return frozendict(join)


def get_best_actions(
    pref: Preference, actions: Set[Trajectory], outcomes: Callable[[Trajectory], PlayerOutcome]
) -> Set[Trajectory]:
    ba_pref: Set[Trajectory] = set(actions)
    for act_1 in frozenset(ba_pref):
        if act_1 not in ba_pref:
            continue
        for act_2 in frozenset(ba_pref):
            if act_1 == act_2:
                continue
            comp: ComparisonOutcome = pref.compare(outcomes(act_1), outcomes(act_2))
            if comp == SECOND_PREFERRED:
                ba_pref.remove(act_1)
                break
            elif comp == FIRST_PREFERRED:
                ba_pref.remove(act_2)
    return ba_pref


def solve_leader_follower(context: LeaderFollowerGameSolvingContext) -> SolvedLeaderFollowerGame:
    lf = context.lf

    if not context.solver_params.use_best_response:
        raise NotImplementedError("Leader follower assumes follower best response!")

    def agg_func(outcomes: List[PlayerOutcome]) -> PlayerOutcome:
        if ac_comp == EXP_ACCOMP:
            agg_outcome = calculate_expectation(outcomes=outcomes)
        elif ac_comp == JOIN_ACCOMP:
            agg_outcome = calculate_join(outcomes=outcomes, pref=lf.pref_leader)
        else:
            raise NotImplementedError(f"Antichain comparison - {ac_comp} not in implemented categories")
        return agg_outcome

    print("\nSolving LeaderFollowerGame:")

    tic1 = perf_counter()
    # Collect all possible actions for players
    lead_actions = set(context.player_actions[lf.leader])
    foll_actions = set(context.player_actions[lf.follower])
    foll_act_1 = next(iter(foll_actions))

    tic = perf_counter()
    # Calculate best responses, corresponding leader outcomes and aggregated leader outcomes
    # -> for every leader action and follower preference
    ac_comp = context.solver_params.antichain_comparison
    br_pref: Dict[Trajectory, Dict[Preference, Set[Trajectory]]] = {}
    agg_out_l: Dict[Trajectory, Dict[Preference, PlayerOutcome]] = {}
    for l_act in lead_actions:
        br_pref[l_act], agg_out_l[l_act] = {}, {}
        joint_act: Dict[PlayerName, Trajectory] = {lf.leader: l_act, lf.follower: foll_act_1}
        for p_f in lf.prefs_follower_est.support():
            _, br = get_best_responses(
                joint_actions=joint_act, context=context, player=lf.follower, done_p=set(), player_pref=p_f
            )

            outcomes_l: List[PlayerOutcome] = []
            # outcomes_f: List[PlayerOutcome] = []
            # act_f: List[Trajectory] = []
            for act in iterate_dict_combinations({lf.leader: {l_act}, lf.follower: br}):
                out = frozendict(context.game_outcomes(act))
                outcomes_l.append(out[lf.leader])
                # outcomes_f.append(out[lf.follower])
                # act_f.append(act[lf.follower])
            agg_out_l[l_act][p_f] = agg_func(outcomes=outcomes_l)
            br_pref[l_act][p_f] = br

    toc = perf_counter() - tic
    print(f"\tBest response time = {toc:.2f} s")

    # Calculate best actions of leader and all possible best actions
    # For every pref combination, compare all agg lead outcomes and select non-dominated ones
    tic = perf_counter()
    best_actions: Dict[Preference, Set[Trajectory]] = {}
    all_actions: Set[Trajectory] = set()
    for p_f in lf.prefs_follower_est.support():

        def get_outcomes(action: Trajectory) -> PlayerOutcome:
            return agg_out_l[action][p_f]

        ba_pref: Set[Trajectory] = get_best_actions(pref=lf.pref_leader, actions=lead_actions, outcomes=get_outcomes)
        best_actions[p_f] = ba_pref
        all_actions |= ba_pref

    # Calculate aggregated outcomes for meet of all prefs (Union of actions)
    # Agg is calc on the previously agg outcome to account for frequency of a given action,
    #   and so the exp would be more accurate (like weighted sum if the same action is played for diff prefs)
    # Join would be the same if done in any order
    agg_meet_l: Dict[Trajectory, PlayerOutcome] = {}
    for l_act in lead_actions:
        outcomes_l: List[PlayerOutcome] = list(agg_out_l[l_act].values())
        agg_meet_l[l_act] = agg_func(outcomes=outcomes_l)

    best_actions_meet = get_best_actions(pref=lf.pref_leader, actions=lead_actions, outcomes=agg_meet_l.__getitem__)
    all_actions |= best_actions_meet

    toc = perf_counter() - tic
    print(f"\tBest leader actions time = {toc:.2f} s")

    # Calculate final outcomes and post-process for data structure
    game_nodes: Dict[Trajectory, Dict[Preference, LeaderFollowerGameNode]] = {}
    for l_act in all_actions:
        pref_nodes: Dict[Preference, LeaderFollowerGameNode] = {}
        for p_f in lf.prefs_follower_est.support():
            solved_game: SolvedTrajectoryGame = set()
            for act in iterate_dict_combinations({lf.leader: {l_act}, lf.follower: br_pref[l_act][p_f]}):
                out = context.game_outcomes(act)
                solved_game.add(SolvedTrajectoryGameNode(actions=act, outcomes=out))
            pref_nodes[p_f] = LeaderFollowerGameNode(nodes=solved_game, agg_lead_outcome=agg_out_l[l_act][p_f])
        game_nodes[l_act] = pref_nodes

    toc = perf_counter() - tic1
    print(f"\tGame solving time = {toc:.2f} s")

    return SolvedLeaderFollowerGame(
        lf=deepcopy(lf), games=game_nodes, best_leader_actions=best_actions, meet_leader_actions=best_actions_meet
    )


def update_follower_prefs(stage: LeaderFollowerGameStage, ps: PossibilityMonad) -> Poss[Preference]:
    lf = stage.lf
    node = stage.game_node
    possible_prefs: Set[Preference] = set()
    for p_f in lf.prefs_follower_est.support():
        _, br = get_best_responses(
            joint_actions=node.actions, context=stage.context, player=lf.follower, done_p=set(), player_pref=p_f
        )
        if node.actions[lf.follower] in br:
            possible_prefs.add(p_f)

    return ps.lift_many(possible_prefs)


def solve_recursive_game_stage(
    game: LeaderFollowerGame, context: LeaderFollowerGameSolvingContext
) -> LeaderFollowerGameStage:
    assert game.lf.pref_follower_real is not None
    # Solve leader game and select action
    lf_game = solve_leader_follower(context=context)
    act_leader = choice(list(lf_game.meet_leader_actions))

    # Select a BR as action for follower
    joint_act = {game.lf.leader: act_leader, game.lf.follower: next(iter(context.player_actions[game.lf.follower]))}

    br_all: Set[Trajectory] = set()
    br: Set[Trajectory] = set()

    def get_br(pref: Preference):
        _, br_pref = get_best_responses(
            joint_actions=joint_act, context=context, player=game.lf.follower, done_p=set(), player_pref=pref
        )
        return br_pref

    for p_f in context.lf.prefs_follower_est.support():
        br_pf = get_br(p_f)
        br_all |= br_pf
        if p_f == game.lf.pref_follower_real:
            br = br_pf
    if len(br) == 0:
        br = get_br(game.lf.pref_follower_real)
    act_follower = choice(list(br))

    # Compute outcomes for both players and save
    joint_act[game.lf.follower] = act_follower
    joint_act_f = frozendict(joint_act)
    outcomes = frozendict(context.game_outcomes(joint_act_f))
    game_node = SolvedTrajectoryGameNode(actions=joint_act_f, outcomes=outcomes)

    # Simulate game forward for one timestep
    states = {pname: player.state for pname, player in game.game_players.items()}
    stage = LeaderFollowerGameStage(
        lf=deepcopy(game.lf),
        context=context,
        lf_game=lf_game,
        game_node=game_node,
        best_responses_pred=br_all,
        states=states,
        time=act_leader.get_start(),
    )
    for pname, state in simulate_recursive_game_stage(game=game, stage=stage).items():
        game.game_players[pname].state = state

    # Update estimate of follower prefs using action as measurement
    if game.lf.update_prefs:
        game.lf.prefs_follower_est = update_follower_prefs(stage=stage, ps=game.ps)
        print(f"Updated Estimated Preferences - remaining = " f"{len(game.lf.prefs_follower_est.support())} types")

    return stage


def simulate_recursive_game_stage(
    game: LeaderFollowerGame, stage: LeaderFollowerGameStage
) -> Mapping[PlayerName, Poss[VehicleState]]:
    states: Dict[PlayerName, Poss[VehicleState]] = {}
    for pname, player in game.game_players.items():
        act = stage.game_node.actions[pname]
        states[pname] = game.ps.unit(act.at(stage.time + game.lf.simulation_step))

    return states


def solve_recursive_game(game: LeaderFollowerGame) -> SolvedRecursiveLeaderFollowerGame:
    tic = perf_counter()
    stage_seq: List[LeaderFollowerGameStage] = []

    done: bool = False
    clearance_dict: Dict[PlayerName, Tuple[SE2Transform, VehicleGeometry]] = {}
    states: Dict[PlayerName, VehicleState] = {}
    for pname, player in game.game_players.items():
        state_p = next(iter(player.state.support()))
        states[pname] = state_p
        clearance_dict[pname] = (Trajectory.state_to_se2(x=state_p), player.vg)

    # Solve all stages of LF game
    for i in range(int(game.lf.solve_time // game.lf.simulation_step)):
        print(f"\n\nRecursive Game: Stage = {i}")
        context: SolvingContext = preprocess_full_game(sgame=game, only_traj=False)
        assert isinstance(context, LeaderFollowerGameSolvingContext)
        stage = solve_recursive_game_stage(game=game, context=context)
        stage_seq.append(stage)
        for pname, player in game.game_players.items():
            for state in game.game_players[pname].state.support():
                clearance_dict[pname] = (Trajectory.state_to_se2(state), player.vg)
                states_p = [states[pname], state]
                if pname == game.lf.leader:
                    for lane, goal in game.world.get_lanes(pname):
                        if Trajectory.trim_trajectory(states=states_p, goal=goal):
                            done = True
                states[pname] = state
        if Clearance.get_clearance(players=clearance_dict) < 1e-3:
            print(f"\n\nCollision detected !!! Stopping")
            break
        if done:
            print(f"\n\nReached terminal progress of lane for leader, stopping.")
            break

    print(f"\n\nCalculating aggregate trajectories and outcomes")
    # Concatenate simulated sections of trajectories to get overall driven trajectories
    times: List[Timestamp] = [stage.time for stage in stage_seq]
    states_traj: Mapping[PlayerName, Dict[Timestamp, VehicleState]] = {pname: {} for pname in game.game_players.keys()}
    lanes: Dict[PlayerName, DgLanelet] = {}
    goals: Dict[PlayerName, Optional[Polygon]] = {}
    for i in range(len(stage_seq)):
        sol = stage_seq[i]
        for pname in game.game_players.keys():
            pact = sol.game_node.actions[pname]
            for step in pact.get_sampling_points():
                if 0 <= step - times[i] <= game.lf.simulation_step:
                    states_traj[pname][step] = pact.at(step)
            lane, goal = pact.get_lane()
            lanes[pname], goals[pname] = lane, goal

    # Trim trajectory of leader till terminal progress and follower till same time
    traj_lead = Trajectory(
        values=list(states_traj[game.lf.leader].values()), lane=lanes[game.lf.leader], goal=goals[game.lf.leader]
    )
    traj_foll = Trajectory(
        values=list(states_traj[game.lf.follower].values())[: len(traj_lead)],
        lane=lanes[game.lf.follower],
        goal=goals[game.lf.follower],
    )
    traj: Mapping[PlayerName, Trajectory] = {game.lf.leader: traj_lead, game.lf.follower: traj_foll}

    # Calculate aggregated outcomes for the driven trajectories
    agg_outcomes = game.get_outcomes(traj)
    l_out = "\n\t".join([str(met) for met in agg_outcomes[game.lf.leader].values()])
    print(f"\nAggregate Outcomes: \n\t{l_out}")
    agg_node = SolvedTrajectoryGameNode(actions=traj, outcomes=agg_outcomes)

    toc = perf_counter() - tic
    print(f"Recursive solution complete. Total time = {toc:.2f} s\n\n")

    return SolvedRecursiveLeaderFollowerGame(
        lf=deepcopy(game.lf), aggregated_node=agg_node, stages=DgSampledSequence(timestamps=times, values=stage_seq)
    )
