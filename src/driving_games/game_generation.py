from dataclasses import dataclass
from decimal import Decimal as D
from typing import cast, Dict, FrozenSet as ASet, Mapping, Tuple

from commonroad.scenario.scenario import Scenario
from cycler import cycler

from dg_commons import PlayerName, fd, fs
from dg_commons.maps import DgLanelet
from dg_commons.sim.models.vehicle_ligths import NO_LIGHTS
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from games import (
    get_accessible_states,
    UncertaintyParams,
)
from possibilities import PossibilityMonad
from .dg_def import DrivingGamePlayer, DrivingGame
from .joint_reward import VehicleJointReward
from .personal_reward import VehiclePersonalRewardStructureTime
from .preferences_coll_time import VehiclePreferencesCollTime
from .structures import (
    VehicleTrackState,
)
from .vehicle_dynamics import VehicleTrackDynamics, VehicleTrackDynamicsParams
from .vehicle_observation import VehicleDirectObservations
from .visualization import DrivingGameVisualization


@dataclass
class DGSimpleParams:
    game_dt: D
    """Game discretization"""
    scenario: Scenario  # fixme maybe a string to be loaded
    """A commonroad scenario"""
    ref_lanes: Mapping[PlayerName, DgLanelet]
    """Reference lanes"""
    progress: Mapping[PlayerName, Tuple[D, D]]
    """Initial and End progress along the reference Lanelet"""
    track_dynamics_param: VehicleTrackDynamicsParams
    """Dynamics the players"""
    shared_resources_ds: D

    def __post__init__(self):
        assert self.ref_lanes.keys() == self.progress.keys()
        for progress in self.progress.values():
            assert progress[0] <= progress[1]


def get_two_vehicle_game(dg_params: DGSimpleParams, uncertainty_params: UncertaintyParams) -> DrivingGame:
    ps: PossibilityMonad = uncertainty_params.poss_monad
    players: Dict[PlayerName, DrivingGamePlayer] = {}
    cc = list(cycler(color=["c", "m", "y", "k"]))

    for i, (p, lane) in enumerate(dg_params.ref_lanes.items()):
        g = VehicleGeometry.default_car(color=cc[i]["color"])
        p_dynamics = VehicleTrackDynamics(
            ref=lane,
            max_path=dg_params.progress[p][1],
            vg=g,
            poss_monad=ps,
            param=dg_params.track_dynamics_param,
        )
        p_init_progress = dg_params.progress[p][0]
        p_ref = lane.lane_pose(float(p_init_progress), 0, 0).center_point
        p_x = VehicleTrackState(
            ref=p_ref, x=p_init_progress, wait=D(0), v=dg_params.track_dynamics_param.min_speed, light=NO_LIGHTS
        )
        p_initial = ps.unit(p_x)
        p_personal_reward_structure = VehiclePersonalRewardStructureTime(goal_progress=dg_params.progress[p][1])
        p_preferences = VehiclePreferencesCollTime()

        # this part about observations is not used at the moment
        g = get_accessible_states(p_initial, p_personal_reward_structure, p_dynamics, dg_params.game_dt)
        p_possible_states = cast(ASet[VehicleTrackState], fs(g.nodes))
        p_observations = VehicleDirectObservations(p_possible_states, {})

        game_p = DrivingGamePlayer(
            initial=p_initial,
            dynamics=p_dynamics,
            observations=p_observations,
            personal_reward_structure=p_personal_reward_structure,
            preferences=p_preferences,
            monadic_preference_builder=uncertainty_params.mpref_builder,
        )
        players.update({p: game_p})
    dt = dg_params.game_dt

    joint_reward = VehicleJointReward(collision_threshold=dg_params.collision_threshold, geometries=geometries)
    game_visualization = DrivingGameVisualization(dg_params, geometries=geometries, ds=dg_params.shared_resources_ds)
    game: DrivingGame = DrivingGame(
        players=fd(players),
        ps=ps,
        joint_reward=joint_reward,
        game_visualization=game_visualization,
    )
    return game
