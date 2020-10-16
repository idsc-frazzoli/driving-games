import itertools
from typing import Mapping, Dict, FrozenSet, Set, Tuple

from frozendict import frozendict
from zuper_commons.types import ZValueError, ZNotImplementedError

from bayesian_driving_games import PlayerType
from bayesian_driving_games.structures_solution import BayesianSolvingContext, BayesianGameNode
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
    JointMixedActions, check_joint_mixed_actions, Game, SR
)
from games.solution_security import get_security_policies
from games.structures_solution import ValueAndActions, STRATEGY_MIX, STRATEGY_SECURITY, STRATEGY_BAIL
from games.utils import fd, valmap
from possibilities import Poss, PossibilityMonad
from possibilities.sets import SetPoss
from preferences import Preference


def weight_outcome(mixed_outcome, weight, ps, t1, t2):
    x = {}
    a = list(mixed_outcome.support())[0]
    for k,v in a.items():
        if (t2,t1) in k:
            x[k] = list(v.support())[0] * weight
    return x


def analyze_sequential_rational(
    *,
    ps: PossibilityMonad,
    gn: BayesianGameNode,
    solved: Dict[JointPureActions, Mapping[Tuple[PlayerName, PlayerType], UncertainCombined]],
    preferences: Mapping[PlayerName, Preference[UncertainCombined]],
    game: Game[X, U, Y, RP, RJ, SR]
) -> EquilibriaAnalysis:
    # Now we want to find all mixed strategies
    # Example: From sets, you could have [A, B] ->  {A}, {B}, {A,B}
    # Example: From probs, you could have [A,B] -> {A:1}, {B:1} , {A:0.5, B:0.5}, ...
    # fixme for probabilities this is restrictive... double check mix method
    player_mixed_strategies: Dict[PlayerName, FrozenSet[Poss[U]]] = valmap(ps.mix, gn.moves)
    # logger.info(player_mixed_strategies=player_mixed_strategies)
    # now we do the product of the mixed strategies
    # let's order them

    # TODO: Not very elegant and prone to errors!
    type_players = []
    helper: str = "init"
    for player_name, player in game.players.items():
        for t in player.types_of_myself:
            helper = "{},{}".format(player_name, t)
            type_players.append(PlayerName(helper))

    players_strategies = {}
    for player_name in game.players:
        for t in type_players:
            if player_name in t:
                players_strategies[t] = player_mixed_strategies[player_name]

    players_ordered = list(players_strategies)  # only the active ones
    players_strategies = [players_strategies[_] for _ in players_ordered]
    all_players = set(gn.states)
    active_players = set(gn.moves)

    player_mixed_strategies_new = dict(zip(type_players, players_strategies))

    _ = next(iter(game.players))
    type_combinations = list(itertools.product(game.players[_].types_of_myself, game.players[_].types_of_other))

    results: Dict[JointMixedActions, Mapping[PlayerName, UncertainCombined]] = {}
    for choices in itertools.product(*tuple(players_strategies)):
        choice: JointMixedActions = frozendict(zip(players_ordered, choices))
        # choice: Mapping[Set[PlayerType], JointMixedActions] = {}
        # for t in type_combinations:
        #     choice[t] = frozendict(zip(players_ordered, choices))
            
        def f(y: JointPureActions) -> JointPureActions:
            return y

        p1 = list(game.players.keys())[0]
        p2 = list(game.players.keys())[1]
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

                a = frozendict(zip([p1,p2], [move1,move2]))
                dist: Poss[JointPureActions] = ps.build_multiple(a=a, f=f)

                mixed_outcome: Poss[Mapping[PlayerName, UncertainCombined]]
                mixed_outcome = ps.build(dist, solved.__getitem__)

                w_outcome = weight_outcome(mixed_outcome, belief, ps, t1, t2)

                for player_name in w_outcome.keys():
                    try:
                        res[player_name[0]] = res[player_name[0]] + w_outcome[player_name]
                    except:
                        res[player_name[0]] = w_outcome[player_name]

        for k,v in res.items():
            res[k] = ps.lift_many([v])

        results[choice] = frozendict(res)

        # preferences_new: Mapping[PlayerName, Preference[UncertainCombined]] = {}
        # for player_name in type_players:
        #     for p_name in preferences.keys():
        #         if p_name in player_name:
        #             preferences_new[player_name] = preferences[p_name]

    # logger.info(results=results)
    return analyze(player_mixed_strategies_new, results, preferences)


def solve_sequential_rationality(
    sc: BayesianSolvingContext,
    gn: BayesianGameNode,
    solved: Dict[JointPureActions, Mapping[Tuple[PlayerName, PlayerType], UncertainCombined]],
) -> ValueAndActions[U, RP, RJ]:
    ps = sc.game.ps
    for pure_action in solved:
        check_joint_pure_actions(pure_action)

    if not gn.moves:
        msg = "Cannot solve_equilibria if there are no moves "
        raise ZValueError(msg, gn=gn)
    # logger.info(gn=gn, solved=solved)
    # logger.info(possibilities=list(solved))
    players_active = set(gn.moves)
    preferences: Dict[PlayerName, Preference[UncertainCombined]]
    preferences = {k: sc.outcome_preferences[k] for k in players_active}

    ea: EquilibriaAnalysis[X, U, Y, RP, RJ]
    ea = analyze_sequential_rational(ps=sc.game.ps, gn=gn, solved=solved, preferences=preferences, game=sc.game)
    players_with_types = list(list(ea.nondom_nash_equilibria.keys())[0].keys())
    # logger.info(ea=ea)
    if len(ea.nondom_nash_equilibria) == 1:

        eq = list(ea.nondom_nash_equilibria)[0]
        check_joint_mixed_actions(eq)

        def f(y: JointPureActions) -> JointPureActions:
            return y

        p1 = list(sc.game.players.keys())[0]
        p2 = list(sc.game.players.keys())[1]
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

                for player_name in sc.game.players:
                    key = (player_name, (t2,t1))
                    x = list(outcome.support())[0]
                    game_value[key] = x[key]

        # game_value = dict(ea.nondom_nash_equilibria[eq])
        for player_final, final_value in gn.is_final.items():
            game_value[player_final] = ps.unit(Combined(final_value, None))
        #if set(game_value) != set(gn.states):
            #raise ZValueError("incomplete", game_value=game_value, gn=gn)
        return ValueAndActions(game_value=frozendict(game_value), mixed_actions=eq)
    else:
        # multiple non-dominated nash equilibria
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
                # check_poss(strategy)
                profile[player_name] = strategy

            def f(y: JointPureActions) -> JointPureActions:
                return frozendict(y)

            dist: Poss[JointPureActions] = ps.build_multiple(a=profile, f=f)

            game_value1: Mapping[PlayerName, UncertainCombined]
            game_value1 = {}
            for player_name in gn.states:

                def f(jpa: JointPureActions) -> UncertainCombined:
                    return solved[jpa][player_name]

                game_value1[player_name] = ps.join(ps.build(dist, f))

            # logger.info(dist=dist)
            # game_value1 = ps.join(ps.build(dist, solved.__getitem__))

            return ValueAndActions(game_value=fd(game_value1), mixed_actions=frozendict(profile))
        # Anything can happen
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


