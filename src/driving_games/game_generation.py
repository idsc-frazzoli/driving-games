from decimal import Decimal as D
from typing import cast, Dict, FrozenSet as ASet

from cycler import cycler

from dg_commons import PlayerName, fd, fs
from dg_commons.sim.models.vehicle_ligths import NO_LIGHTS
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from driving_games.dg_def import DrivingGamePlayer, DrivingGame, DGSimpleParams
from driving_games.joint_reward import VehicleJointReward
from driving_games.personal_reward import VehiclePersonalRewardStructureTime
from driving_games.preferences_coll_time import VehiclePreferencesCollTime
from driving_games.structures import VehicleTrackState
from driving_games.vehicle_dynamics import VehicleTrackDynamics
from driving_games.vehicle_observation import VehicleDirectObservations
from driving_games.visualization import DrivingGameVisualization
from games import (
    get_accessible_states,
    UncertaintyParams,
)
from possibilities import PossibilityMonad

__all__ = ["initialize_driving_game"]


def initialize_driving_game(dg_params: DGSimpleParams, uncertainty_params: UncertaintyParams) -> DrivingGame:
    ps: PossibilityMonad = uncertainty_params.poss_monad
    players: Dict[PlayerName, DrivingGamePlayer] = {}
    geometries: Dict[PlayerName, VehicleGeometry] = {}
    cc = list(cycler(color=["c", "m", "y", "k"]))

    for i, (p, lane) in enumerate(dg_params.ref_lanes.items()):
        g = VehicleGeometry.default_car(color=cc[i]["color"])
        geometries[p] = g
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
    joint_reward = VehicleJointReward(game_dt=dt, geometries=geometries, col_check_dt=0.4)
    game_visualization = DrivingGameVisualization(
        dg_params, geometries=geometries, ds=dg_params.shared_resources_ds, plot_limits=dg_params.plot_limits
    )

    game: DrivingGame = DrivingGame(
        players=fd(players),
        ps=ps,
        joint_reward=joint_reward,
        game_visualization=game_visualization,
    )
    return game
