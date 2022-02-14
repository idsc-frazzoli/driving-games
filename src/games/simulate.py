from dataclasses import dataclass
from decimal import Decimal as D
from typing import Generic, Mapping, Optional, TypeVar

from frozendict import frozendict

from dg_commons import (
    PlayerName,
    RJ,
    RP,
    U,
    X,
    Y,
    DgSampledSequence,
    DgSampledSequenceBuilder as DgSSBuilder,
    Timestamp,
)
from .checks import check_joint_pure_actions
from .game_def import (
    AgentBelief,
    Game,
    JointPureActions,
    JointState,
    SR,
)

__all__ = ['Simulation']


@dataclass
class SimulationStep(Generic[X, U, Y, RP, RJ]):
    states: JointState
    pure_actions: JointPureActions
    incremental_costs: Mapping[PlayerName, RP]
    joint_cost: Optional[RJ]

    def __post_init__(self) -> None:
        check_joint_pure_actions(self.pure_actions)


@dataclass
class Simulation(Generic[X, U, Y, RP, RJ]):
    states: DgSampledSequence[JointState]
    actions: DgSampledSequence[JointPureActions]
    costs: DgSampledSequence[Mapping[PlayerName, RP]]
    joint_costs: DgSampledSequence[Mapping[PlayerName, RJ]]


N = TypeVar("N")


def simulate1(
    game: Game[X, U, Y, RP, RJ, SR],
    policies: Mapping[PlayerName, AgentBelief[X, U]],
    initial_states: JointState,
    dt: D,
    seed: int,
) -> Simulation[X, U, Y, RP, RJ]:
    S_states: DgSSBuilder[JointState] = DgSSBuilder[JointState]()
    S_actions: DgSSBuilder[JointPureActions] = DgSSBuilder[JointPureActions]()
    S_costs: DgSSBuilder[Mapping[PlayerName, RP]] = DgSSBuilder[Mapping[PlayerName, RP]]()
    S_joint_costs: DgSSBuilder[Mapping[PlayerName, RJ]] = DgSSBuilder[Mapping[PlayerName, RJ]]()

    S_states.add(D(0), initial_states)
    ps = game.ps
    sampler = ps.get_sampler(seed)

    while True:
        # last time and state
        t1: Timestamp = S_states.timestamps[-1]
        s1: JointState = S_states.values[-1]
        players_active = set(s1)

        if not players_active:
            break

        if game.joint_reward.is_joint_final_states(s1):
            # this is not okay for solutions that do not terminate for everyone
            S_joint_costs.add(t1, game.joint_reward.joint_final_reward(s1))
            break

        s1_actions = {}
        next_states = {}
        personal_costs = {}
        for player_name in players_active:
            state_self = s1[player_name]
            player = game.players[player_name]
            prs = player.personal_reward_structure
            is_final = prs.is_personal_final_state(state_self)
            if is_final:  # no actions for him
                personal_costs[player_name] = prs.personal_final_reward(state_self)
                continue

            policy = policies[player_name]

            state_others = frozendict({k: v for k, v in s1.items() if k != player_name})
            belief_state_others = ps.unit(state_others)

            p_action = policy.get_commands(state_self, belief_state_others)

            action = sampler.sample(p_action)
            personal_costs[player_name] = prs.personal_reward_incremental(state_self, action, dt)

            dynamics = game.players[player_name].dynamics
            state_player = s1[player_name]
            action_to_successors = dynamics.successors(state_player, dt)
            succ = action_to_successors[action]
            next_state = sampler.sample(succ)

            s1_actions[player_name] = action
            next_states[player_name] = next_state

        inc_joint: Mapping[PlayerName, RJ]
        transitions = {
            p: DgSampledSequence[X](timestamps=(D(0), dt), values=(s1[p], next_states[p])) for p in
            next_states
        }
        inc_joint = game.joint_reward.joint_reward_incremental(txs=transitions)
        S_joint_costs.add(t1, inc_joint)

        S_actions.add(t1, frozendict(s1_actions))
        S_costs.add(t1, frozendict(personal_costs))
        t2 = t1 + dt
        S_states.add(t2, frozendict(next_states))

    return Simulation(
        states=S_states.as_sequence(),
        actions=S_actions.as_sequence(),
        costs=S_costs.as_sequence(),
        joint_costs=S_joint_costs.as_sequence(),
    )
