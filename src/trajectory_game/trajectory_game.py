from itertools import product
from typing import Dict, Set, FrozenSet, Mapping, Tuple
from frozendict import frozendict
from time import perf_counter

from games import PlayerName
from possibilities import Poss
from preferences import Preference, ComparisonOutcome, FIRST_PREFERRED, INCOMPARABLE, INDIFFERENT, SECOND_PREFERRED

from .structures import VehicleState
from .paths import Trajectory
from .world import World
from .metrics_def import PlayerOutcome, TrajectoryGameOutcome
from .game_def import StaticGame, StaticGamePlayer, StaticSolvingContext, StaticSolvedGameNode

__all__ = [
    "JointTrajProfile",
    "TrajectoryGamePlayer",
    "TrajectoryGame",
    "SolvedTrajectoryGameNode",
    "SolvedTrajectoryGame",
    "compute_solving_context",
    "solve_game",
]

JointTrajSet = Mapping[PlayerName, FrozenSet[Trajectory]]
JointTrajProfile = Mapping[PlayerName, Trajectory]


class TrajectoryGamePlayer(StaticGamePlayer[VehicleState, Trajectory, World, PlayerOutcome]):
    pass


class TrajectoryGame(StaticGame[VehicleState, Trajectory, World, PlayerOutcome]):
    pass


class SolvedTrajectoryGameNode(StaticSolvedGameNode[Trajectory, PlayerOutcome]):
    pass


SolvedTrajectoryGame = Set[SolvedTrajectoryGameNode]


def get_joint_traj(actions: JointTrajSet) -> JointTrajProfile:
    """Compute the cartesian product of actions for all players"""
    for joint_traj in (frozendict(zip(actions.keys(), x)) for x in product(*actions.values())):
        yield joint_traj


def compute_solving_context(traj_game: StaticGame) -> \
        StaticSolvingContext:
    """
    Preprocess the game -> Compute all possible actions and outcomes for each combination
    """

    # Generate the trajectories for each player
    all_traj: Dict[PlayerName, Poss[FrozenSet[Trajectory]]] = {}
    for player_name, game_player in traj_game.game_players.items():
        def get_traj_set(state: VehicleState) -> FrozenSet[Trajectory]:
            return game_player.action_set_generator.get_action_set(state=state,
                                                                   world=traj_game.world,
                                                                   player=player_name)

        all_traj[player_name] = traj_game.ps.build(a=game_player.state, f=get_traj_set)

    def build_sets(data_in: JointTrajSet) -> Mapping[JointTrajProfile, TrajectoryGameOutcome]:
        ret: Dict[JointTrajProfile, TrajectoryGameOutcome] = {}
        for joint_traj in get_joint_traj(data_in):
            ret[joint_traj] = traj_game.game_outcomes(joint_traj, traj_game.world)
        return frozendict(ret)

    tic = perf_counter()
    traj_outcomes = traj_game.ps.build_multiple(a=all_traj, f=build_sets)
    toc = perf_counter() - tic
    print('Outcomes evaluation time = {} s'.format(toc))

    # return traj_outcomes for Poss[Mapping[JointTrajProfile, TrajectoryGameOutcome]]

    # TODO[SIR]: Extend to poss. Convert poss to single value for now
    outcomes: Mapping[JointTrajProfile, TrajectoryGameOutcome] = list(traj_outcomes.support())[0]

    actions: Dict[PlayerName, FrozenSet[Trajectory]] = {}
    for player_name, poss in all_traj.items():
        actions[player_name] = list(poss.support())[0]

    # Similar to get_outcome_preferences_for_players, use SetPreference1 for Poss
    pref: Mapping[PlayerName, Preference[PlayerOutcome]] = \
        {name: player.preferences for name, player in traj_game.game_players.items()}

    context = StaticSolvingContext(player_actions=actions, game_outcomes=outcomes,
                                   outcome_pref=pref)
    return context


def solve_game(context: StaticSolvingContext) ->\
        Tuple[SolvedTrajectoryGame, SolvedTrajectoryGame,
              SolvedTrajectoryGame, SolvedTrajectoryGame]:

    indiff_nash: SolvedTrajectoryGame = set()
    incomp_nash: SolvedTrajectoryGame = set()
    weak_nash: SolvedTrajectoryGame = set()
    strong_nash: SolvedTrajectoryGame = set()

    def get_action_options(joint_act: JointTrajProfile, pname: PlayerName) -> JointTrajSet:
        """Returns all possible actions for the player, with other player actions frozen
           Current player action is not included"""

        def get_actions(name: PlayerName) -> FrozenSet[Trajectory]:
            if name == pname:
                p_actions: Set[Trajectory] = set(actions[name])
                p_actions.remove(joint_act[pname])
                return frozenset(p_actions)
            return frozenset({joint_act[name]})

        ret: JointTrajSet = {_: get_actions(_) for _ in players}
        return ret

    def get_solved_game_node(act: JointTrajProfile, out: TrajectoryGameOutcome) -> \
            SolvedTrajectoryGameNode:
        return SolvedTrajectoryGameNode(actions=act, outcomes=out)

    # TODO[SIR]: Remove dominated options first or just brute force through?
    actions = context.player_actions
    players = actions.keys()

    tic = perf_counter()
    # For each possible action combination, check if it is a nash eq
    for joint_actions in get_joint_traj(actions):

        # For each player, compare the current outcome to their alternatives
        # Classify into types of nash eq. based on the outputs
        outcome: TrajectoryGameOutcome = context.game_outcomes[joint_actions]
        results: Set[ComparisonOutcome] = set()

        for player in players:
            action_alt: JointTrajSet = get_action_options(joint_act=joint_actions, pname=player)
            player_outcome: PlayerOutcome = outcome[player]
            for joint_act_alt in get_joint_traj(action_alt):
                player_outcome_alt: PlayerOutcome = \
                    context.game_outcomes[joint_act_alt][player]
                comp_outcome: ComparisonOutcome = \
                    context.outcome_pref[player].compare(player_outcome, player_outcome_alt)
                results.add(comp_outcome)

                # If second option is preferred, current point is not a nash eq.
                if comp_outcome == SECOND_PREFERRED:
                    break

            # If second option is preferred for any player, current point is not a nash eq.
            if SECOND_PREFERRED in results:
                break

        # If second option is preferred for any player, current point is not a nash eq.
        if SECOND_PREFERRED in results:
            continue

        solved_node: SolvedTrajectoryGameNode = \
            get_solved_game_node(act=joint_actions, out=outcome)
        if results == {FIRST_PREFERRED}:
            strong_nash.add(solved_node)
            continue
        if INDIFFERENT in results and INCOMPARABLE in results:
            weak_nash.add(solved_node)
            indiff_nash.add(solved_node)
            incomp_nash.add(solved_node)
        elif INDIFFERENT in results:
            indiff_nash.add(solved_node)
            weak_nash.add(solved_node)
        elif INCOMPARABLE in results:
            incomp_nash.add(solved_node)
            weak_nash.add(solved_node)

    toc = perf_counter() - tic
    print('Nash equilibrium computation time = {} s'.format(toc))

    return indiff_nash, incomp_nash, weak_nash, strong_nash
