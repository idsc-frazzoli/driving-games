import itertools
from typing import Mapping, Dict, FrozenSet, Tuple, List

from frozendict import frozendict
from zuper_commons.types import ZValueError, ZNotImplementedError

from bayesian_driving_games.structures import PlayerType
from bayesian_driving_games.structures_solution import BayesianGameNode
from games import JointPureActions, PlayerName
from games.equilibria import EquilibriaAnalysis, analyze
from games.game_def import (
    UncertainCombined,
    U,
    RP,
    RJ,
    check_joint_pure_actions,
    X,
    Y,
    Combined,
    JointMixedActions,
    check_joint_mixed_actions,
    Game,
    SR,
)
from games.solution_security import get_security_policies
from games.structures_solution import (
    ValueAndActions,
    STRATEGY_MIX,
    STRATEGY_SECURITY,
    STRATEGY_BAIL,
    SolvingContext,
)
from games.utils import fd, valmap
from possibilities import Poss, PossibilityMonad
from preferences import Preference


def weight_outcome(mixed_outcome, weight, ps, t1, t2):
    """
    This function takes a belief to weigh the mixed_outcome of an action profile to make it an expected value outcome.

    :param mixed_outcome:
    :param weight: A weight (in this case a belief)
    :param ps: The prability monad used in the game
    :param t1: A type
    :param t2: A type
    :return: The weighted outcome
    """
    x = {}
    a = list(mixed_outcome.support())[0]
    for k, v in a.items():
        if (t2, t1) in k:
            x[k] = list(v.support())[0] * weight
        elif (t1, t2) in k:
            x[k] = list(v.support())[0] * weight
    return x


def analyze_sequential_rational(
    *,
    ps: PossibilityMonad,
    gn: BayesianGameNode,
    solved: Dict[JointPureActions, Mapping[Tuple[PlayerName, PlayerType], UncertainCombined]],
    preferences: Mapping[PlayerName, Preference[UncertainCombined]],
    game: Game[X, U, Y, RP, RJ, SR],
) -> EquilibriaAnalysis:
    """
    For each node that is not final, this function is used. It selects an action for each type of each player and
    combines them to an action profile. This action profile and the beliefs are then used to figure out the expected
    value of the action profile. The results of every possible action combination (note an action for each player in
    each type) are given to the analyze function, that determines the Nash equilibrium.

    :param ps: Possibility monad of the game
    :param gn: A Bayesian game node
    :param solved: For each action profile possible at the current state, the expected rewards for each type combination
    :param preferences:
    :param game:
    :return: all Nash Equilibria and the set of dominated Nash Equilibria
    """
    # Now we want to find all mixed strategies
    # Example: From sets, you could have [A, B] ->  {A}, {B}, {A,B}
    # Example: From probs, you could have [A,B] -> {A:1}, {B:1} , {A:0.5, B:0.5}, ...
    # fixme for probabilities this is restrictive... double check mix method
    player_mixed_strategies: Dict[PlayerName, FrozenSet[Poss[U]]] = valmap(ps.mix, gn.moves)

    # TODO: Not very elegant and prone to errors!
    type_players = []
    helper: str = "init"
    for player_name, player in game.players.items():
        for t in player.types_of_myself:
            helper = "{},{}".format(player_name, t)
            type_players.append(PlayerName(helper))

    players_strategies = {}
    for player_name in player_mixed_strategies.keys():
        for t in type_players:
            if player_name in t:
                players_strategies[t] = player_mixed_strategies[player_name]

    players_ordered = list(players_strategies)  # only the active ones
    players_strategies = [players_strategies[_] for _ in players_ordered]
    all_players = set(gn.states)
    active_players = set(gn.moves)

    player_mixed_strategies_new = dict(zip(players_ordered, players_strategies))

    _ = next(iter(game.players))
    type_combinations = list(
        itertools.product(game.players[_].types_of_myself, game.players[_].types_of_other)
    )

    results: Dict[JointMixedActions, Mapping[PlayerName, UncertainCombined]] = {}
    for choices in itertools.product(*tuple(players_strategies)):
        choice: JointMixedActions = frozendict(zip(players_ordered, choices))

        def f(y: JointPureActions) -> JointPureActions:
            return y

        # TODO: Has to be changed in order for >2 players to work.
        p1 = list(player_mixed_strategies)[0]
        p2 = list(player_mixed_strategies)[-1]
        # Get the expected values here
        if p1 == p2:
            res: Dict[PlayerName, UncertainCombined] = {}
            for t1 in game.players[p1].types_of_other:
                for t2 in game.players[p2].types_of_myself:
                    belief = gn.game_node_belief[p1].support()[t1]
                    key1 = "{},{}".format(p1, t2)
                    move1 = choice[key1]
                    a = frozendict(zip([p1], [move1]))
                    dist: Poss[JointPureActions] = ps.build_multiple(a=a, f=f)
                    mixed_outcome: Poss[Mapping[PlayerName, UncertainCombined]]
                    mixed_outcome = ps.build(dist, solved.__getitem__)
                    w_outcome = weight_outcome(mixed_outcome, belief, ps, t1, t2)
                    for player_name in w_outcome.keys():
                        try:
                            res[player_name[0]] = res[player_name[0]] + w_outcome[player_name]
                        except:
                            res[player_name[0]] = w_outcome[player_name]
            for k, v in res.items():
                res[k] = ps.lift_many([v])

            results[choice] = frozendict(res)

        else:
            res: Dict[PlayerName, UncertainCombined] = {}
            for t1 in game.players[p1].types_of_other:
                for t2 in game.players[p2].types_of_other:
                    belief1 = gn.game_node_belief[p1].support()[t1]
                    belief2 = gn.game_node_belief[p2].support()[t2]
                    belief = belief1 * belief2

                    key1 = "{},{}".format(p1, t2)
                    key2 = "{},{}".format(p2, t1)

                    move1 = choice[key1]
                    move2 = choice[key2]

                    a = frozendict(zip([p1, p2], [move1, move2]))
                    dist: Poss[JointPureActions] = ps.build_multiple(a=a, f=f)

                    mixed_outcome: Poss[Mapping[PlayerName, UncertainCombined]]
                    mixed_outcome = ps.build(dist, solved.__getitem__)

                    w_outcome = weight_outcome(mixed_outcome, belief, ps, t1, t2)

                    for player_name in w_outcome.keys():
                        try:
                            res[player_name[0]] = res[player_name[0]] + w_outcome[player_name]
                        except:
                            res[player_name[0]] = w_outcome[player_name]

            for k, v in res.items():
                res[k] = ps.lift_many([v])

            results[choice] = frozendict(res)

    return analyze(player_mixed_strategies_new, results, preferences)


def solve_sequential_rationality(
    sc: SolvingContext,
    gn: BayesianGameNode,
    solved: Dict[JointPureActions, Mapping[Tuple[PlayerName, PlayerType], UncertainCombined]],
) -> ValueAndActions[U, RP, RJ]:
    """
    Uses the analyze_sequential_rational function and then selects if there are multiple equilibria. Selects for each
    player in each type a action.

    :param sc: Game parameters etc.
    :param gn: The current Bayesian game node.
    :param solved: For each action profile, the expected results.
    :return: a value and action of the current game node
    """
    ps = sc.game.ps
    for pure_action in solved:
        check_joint_pure_actions(pure_action)

    if not gn.moves:
        msg = "Cannot solve_equilibria if there are no moves "
        raise ZValueError(msg, gn=gn)

    players_active = set(gn.moves)
    preferences: Dict[PlayerName, Preference[UncertainCombined]]
    preferences = {k: sc.outcome_preferences[k] for k in players_active}

    ea: EquilibriaAnalysis[X, U, Y, RP, RJ]
    ea = analyze_sequential_rational(
        ps=sc.game.ps, gn=gn, solved=solved, preferences=preferences, game=sc.game
    )
    try:
        players_with_types = list(list(ea.nondom_nash_equilibria.keys())[0].keys())
    except:
        print("could not find a pure strategy Nash equuilibrium")

    if len(ea.nondom_nash_equilibria) == 1:

        eq = list(ea.nondom_nash_equilibria)[0]
        check_joint_mixed_actions(eq)

        def f(y: JointPureActions) -> JointPureActions:
            return y

        p1 = list(players_active)[0]
        p2 = list(players_active)[-1]
        if p1 == p2:
            game_value = {}
            for t1 in sc.game.players[p1].types_of_myself:
                for t2 in sc.game.players[p1].types_of_other:
                    key1 = "{},{}".format(p1, t1)
                    move1 = eq[key1]
                    a = frozendict(zip([p1], [move1]))
                    dist: Poss[JointPureActions] = ps.build_multiple(a=a, f=f)

                    outcome: Poss[Mapping[PlayerName, UncertainCombined]]
                    outcome = ps.build(dist, solved.__getitem__)

                    x = list(outcome.support())[0]
                    for key in x.keys():
                        game_value[key] = x[key]

        else:
            game_value = {}
            for t1 in sc.game.players[p1].types_of_other:
                for t2 in sc.game.players[p2].types_of_other:

                    key1 = "{},{}".format(p1, t2)
                    key2 = "{},{}".format(p2, t1)

                    move1 = eq[key1]
                    move2 = eq[key2]

                    a = frozendict(zip([p1, p2], [move1, move2]))
                    dist: Poss[JointPureActions] = ps.build_multiple(a=a, f=f)

                    outcome: Poss[Mapping[PlayerName, UncertainCombined]]
                    outcome = ps.build(dist, solved.__getitem__)

                    x = list(outcome.support())[0]
                    for key in x.keys():
                        game_value[key] = x[key]

        for player_final, final_value in gn.is_final.items():
            for tc, fv in final_value.items():
                game_value[player_final, tc] = ps.unit(Combined(fv, None))
        return ValueAndActions(game_value=frozendict(game_value), mixed_actions=eq)
    else:
        outcomes = set(ea.nondom_nash_equilibria.values())

        strategy = sc.solver_params.strategy_multiple_nash
        if strategy == STRATEGY_MIX:
            # fixme: Not really sure this makes sense when there are probabilities
            profile: Dict[PlayerName, Poss[U]] = {}
            for player_name in players_with_types:
                # find all the mixed strategies he would play at equilibria
                res = set()
                for _ in ea.nondom_nash_equilibria:
                    res.add(_[player_name])
                strategy = ps.join(ps.lift_many(res))
                profile[player_name] = strategy

            def f(y: JointPureActions) -> JointPureActions:
                return frozendict(y)

            p1 = list(players_active)[0]
            p2 = list(players_active)[-1]
            if p1 == p2:  # this loop has to be here because of the preprocessing.
                game_value1: Mapping[PlayerName, UncertainCombined]
                game_value1 = {}

                for t1 in sc.game.players[p1].types_of_myself:
                    for t2 in sc.game.players[p1].types_of_other:
                        outcome: List[Poss[Mapping[Tuple[PlayerName, PlayerType], UncertainCombined]]] = []
                        for eq in list(ea.nondom_nash_equilibria):
                            key1 = "{},{}".format(p1, t1)
                            move1 = eq[key1]
                            a = frozendict(zip([p1], [move1]))
                            dist: Poss[JointPureActions] = ps.build_multiple(a=a, f=f)

                            outcome.append(ps.build(dist, solved.__getitem__))

                        gv = ps.join(ps.lift_many(outcome))

                        x = list(gv.support())[0]
                        for key in x.keys():
                            game_value1[key] = x[key]
            else:

                game_value1: Mapping[PlayerName, UncertainCombined]
                game_value1 = {}

                for t1 in sc.game.players[p1].types_of_other:
                    for t2 in sc.game.players[p2].types_of_other:
                        outcome: List[Poss[Mapping[Tuple[PlayerName, PlayerType], UncertainCombined]]] = []
                        for eq in list(ea.nondom_nash_equilibria):

                            key1 = "{},{}".format(p1, t2)
                            key2 = "{},{}".format(p2, t1)

                            move1 = eq[key1]
                            move2 = eq[key2]

                            a = frozendict(zip([p1, p2], [move1, move2]))
                            dist: Poss[JointPureActions] = ps.build_multiple(a=a, f=f)

                            outcome.append(ps.build(dist, solved.__getitem__))

                        gv = ps.join(ps.lift_many(outcome))

                        x = list(gv.support())[0]
                        for key in x.keys():
                            game_value1[key] = x[key]

            for player_final, final_value in gn.is_final.items():
                for tc, fv in final_value.items():
                    game_value1[player_final, tc] = ps.unit(Combined(fv, None))
            return ValueAndActions(game_value=fd(game_value1), mixed_actions=frozendict(profile))
        # Anything can happen
        # TODO: Not yet updated to Bayesian Games!
        elif strategy == STRATEGY_SECURITY:
            security_policies: JointMixedActions
            security_policies = get_security_policies(ps, solved, sc.outcome_preferences, ea)
            check_joint_mixed_actions2(security_policies)
            dist: Poss[JointPureActions]
            dist = get_mixed2(ps, security_policies)
            # logger.info(dist=dist)
            for _ in dist.support():
                check_joint_pure_actions(_)
            # logger.info(dist=dist)
            game_value = {}
            for player_name in gn.states:

                def f(jpa: JointPureActions) -> UncertainCombined:
                    return solved[jpa][player_name]

                game_value[player_name] = ps.join(ps.build(dist, f))

            # game_value: Mapping[PlayerName, UncertainCombined]
            game_value_ = fd(game_value)
            return ValueAndActions(game_value=game_value_, mixed_actions=security_policies)
        elif strategy == STRATEGY_BAIL:
            msg = "Multiple Nash Equilibria"
            raise ZNotImplementedError(msg, ea=ea)
        else:
            assert False, strategy
