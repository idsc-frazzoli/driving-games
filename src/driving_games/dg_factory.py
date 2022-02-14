from decimal import Decimal as D
from typing import cast, Dict, FrozenSet as ASet

from cycler import cycler

from dg_commons import PlayerName, fd, fs
from dg_commons.sim.models.vehicle_ligths import NO_LIGHTS
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from driving_games.dg_def import DrivingGamePlayer, DrivingGame, DgSimpleParams
from driving_games.preferences_coll_time import VehiclePreferencesCollTime
from driving_games.reward_joint import VehicleJointReward
from driving_games.reward_personal import VehiclePersonalRewardStructureTime
from driving_games.structures import VehicleTrackState
from driving_games.vehicle_dynamics import VehicleTrackDynamics
from driving_games.vehicle_observation import VehicleDirectObservations
from driving_games.visualization import DrivingGameVisualization
from games import UncertaintyParams
from games.preprocess import get_reachable_states
from possibilities import PossibilityMonad

__all__ = ["get_driving_game"]


def get_driving_game(dg_params: DgSimpleParams, uncertainty_params: UncertaintyParams) -> DrivingGame:
    ps: PossibilityMonad = uncertainty_params.poss_monad
    players: Dict[PlayerName, DrivingGamePlayer] = {}
    geometries: Dict[PlayerName, VehicleGeometry] = {}
    cc = list(cycler(color=["c", "m", "y", "gray", "b", "g", "r"]))

    for i, (p, lane) in enumerate(dg_params.ref_lanes.items()):
        g = VehicleGeometry.default_car(color=cc[i]["color"], w_half=0.8)
        geometries[p] = g
        p_dynamics = VehicleTrackDynamics(
            ref=lane,
            vg=g,
            poss_monad=ps,
            param=dg_params.track_dynamics_param,
        )
        p_init_progress = dg_params.progress[p][0]
        # p_ref = lane.lane_pose(float(p_init_progress), 0, 0).center_point
        p_x = VehicleTrackState(
            x=p_init_progress,
            v=dg_params.track_dynamics_param.min_speed + D(1),
            wait=D(0),
            light=NO_LIGHTS,
            has_collided=False,
        )
        p_initial = ps.unit(p_x)
        p_personal_reward_structure = VehiclePersonalRewardStructureTime(goal_progress=dg_params.progress[p][1])
        p_preferences = VehiclePreferencesCollTime()

        # this part about observations is not used at the moment
        # g = get_reachable_states(p_initial, p_personal_reward_structure, p_dynamics, D("1"))
        # # fixme if discretization is a parameter of the solver here it should not depend on it
        # p_possible_states = cast(ASet[VehicleTrackState], fs(g.nodes))
        # p_observations = VehicleDirectObservations(p_possible_states, {})

        game_p = DrivingGamePlayer(
            initial=p_initial,
            dynamics=p_dynamics,
            observations=None,
            personal_reward_structure=p_personal_reward_structure,
            preferences=p_preferences,
            monadic_preference_builder=uncertainty_params.mpref_builder,
        )
        players.update({p: game_p})
    joint_reward = VehicleJointReward(
        geometries=geometries,
        ref_lanes=dg_params.ref_lanes,
        col_check_dt=dg_params.col_check_dt,
        lanelet_network=dg_params.scenario.lanelet_network,
        min_safety_distance=dg_params.min_safety_distance,
    )
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
