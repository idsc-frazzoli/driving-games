from dataclasses import dataclass
from decimal import Decimal as D
from typing import cast, Dict, FrozenSet as ASet, Mapping

from commonroad.scenario.scenario import Scenario

from dg_commons import PlayerName, fd, fs
from dg_commons.maps import DgLanelet
from dg_commons.sim.models.vehicle_ligths import NO_LIGHTS
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from games import (
    GameVisualization,
    get_accessible_states,
    UncertaintyParams,
)
from possibilities import PossibilityMonad
from .dg_def import DrivingGamePlayer, DrivingGame
from .joint_reward import VehicleJointReward
from .personal_reward import VehiclePersonalRewardStructureTime
from .preferences_coll_time import VehiclePreferencesCollTime
from .structures import (
    VehicleActions,
    VehicleCosts,
    VehicleState,
)
from .vehicle_dynamics import VehicleTrackDynamics, VehicleTrackDynamicsParams
from .vehicle_observation import VehicleDirectObservations, VehicleObs
from .visualization import DrivingGameVisualization


@dataclass
class DGSimpleParams:
    game_dt: D
    """Game discretization"""
    scenario: Scenario
    """A commonroad scenario"""
    ref_lanes: Mapping[PlayerName, DgLanelet]
    """Reference lanes"""
    initial_progress: Mapping[PlayerName, D]
    """Initial progress along the reference Lanelet"""
    end_progress: Mapping[PlayerName, D]
    """Goal progress along reference that ends the game"""
    track_dynamics_param: VehicleTrackDynamicsParams
    """Dynamics the players"""
    shared_resources_ds: D

    def __post__init__(self):
        assert self.ref_lanes.keys() == self.initial_progress.keys() == self.end_progress.keys()


def get_two_vehicle_game(dg_params: DGSimpleParams, uncertainty_params: UncertaintyParams) -> DrivingGame:
    ps: PossibilityMonad = uncertainty_params.poss_monad
    players: Dict[PlayerName, DrivingGamePlayer] = {}

    for p, lane in dg_params.ref_lanes.items():
        g = VehicleGeometry.default_car(color=(1, 0, 0))  # todo fix color iterator
        p_dynamics = VehicleTrackDynamics(
            ref=lane,
            max_path=dg_params.end_progress[p],
            vg=g,
            poss_monad=ps,
            param=dg_params.track_dynamics_param,
        )
        p_init_progress = dg_params.initial_progress[p]
        p_ref = lane.lane_pose(float(p_init_progress), 0, 0).center_point
        p_x = VehicleState(
            ref=p_ref, x=p_init_progress, wait=D(0), v=dg_params.track_dynamics_param.min_speed, light=NO_LIGHTS
        )
        p_initial = ps.unit(p_x)
        p_personal_reward_structure = VehiclePersonalRewardStructureTime(goal_progress=dg_params.end_progress[p])
        p_preferences = VehiclePreferencesCollTime()

        # this part about observations is not used at the moment
        g = get_accessible_states(p_initial, p_personal_reward_structure, p_dynamics, dg_params.game_dt)
        p_possible_states = cast(ASet[VehicleState], fs(g.nodes))
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
