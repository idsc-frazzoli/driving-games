from dataclasses import dataclass
from decimal import Decimal as D
from typing import Dict, Generic, Mapping, Optional

from frozendict import frozendict

from .game_def import AgentBelief, Game, PlayerName, RJ, RP, U, X, Y


@dataclass
class SimulationStep(Generic[X, U, Y, RP, RJ]):
    states: Mapping[PlayerName, X]
    actions: Mapping[PlayerName, U]
    incremental_costs: Mapping[PlayerName, RP]
    joint_cost: Optional[RJ]


@dataclass
class Simulation(Generic[X, U, Y, RP, RJ]):
    states: Mapping[D, Mapping[PlayerName, X]]
    actions: Mapping[D, Mapping[PlayerName, U]]
    costs: Mapping[D, Mapping[PlayerName, RP]]
    joint_costs: Mapping[D, Mapping[PlayerName, RJ]]


def simulate1(game: Game[X, U, Y, RP, RJ],
                          policies: Mapping[PlayerName, AgentBelief[X, U]],
                          initial_states: Mapping[PlayerName, X],
                          dt: D) -> Simulation[X, U, Y, RP, RJ]:
    S_states: Dict[D, Mapping[PlayerName, X]] = {}
    S_actions: Dict[D, Mapping[PlayerName, X]] = {}
    S_costs: Dict[D, Mapping[PlayerName, RP]] = {}
    S_joint_costs:Dict[D, Mapping[PlayerName, RJ]] = {}

    S_states[D(0)] = initial_states


    while True:
        # last time
        t1 = list(S_states)[-1]
        # last step
        s1 = S_states[t1]

        players_active = set(s1)
        if not players_active:
            break

        if game.joint_reward.is_joint_final_state(s1):
            S_joint_costs[t1] = game.joint_reward.joint_reward(s1)
            break

        s1_actions = {}
        next_states = {}
        incremental_costs = {}
        for player_name in players_active:
            state_self = s1[player_name]
            player = game.players[player_name]
            prs = player.personal_reward_structure
            is_final = prs.is_personal_final_state(state_self)
            if is_final:
                # no actions for him
                incremental_costs[player_name] = prs.personal_final_reward(state_self)
                continue

            policy = policies[player_name]

            belief_state_others = {k: frozenset({v}) for k, v in s1.items() if k != player_name}
            action = policy.get_commands(state_self, belief_state_others)
            incremental_costs[player_name] = prs.personal_reward_incremental(state_self, action, dt)


            dynamics = game.players[player_name].dynamics
            successors = dynamics.successors(s1[player_name], dt)[action]
            next_state = list(successors)[0]

            s1_actions[player_name] = action
            next_states[player_name]  = next_state


        S_actions[t1] = frozendict(s1_actions)
        S_costs[t1] = frozendict(incremental_costs)
        t2 = t1 + dt
        S_states[t2] = frozendict(next_states)


    return Simulation(states=frozendict(S_states), actions=frozendict(S_actions),
                      costs=frozendict(S_costs), joint_costs=frozendict(S_joint_costs))
