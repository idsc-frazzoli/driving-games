from decimal import Decimal as D
from typing import Dict

from cycler import cycler

from dg_commons import fd, PlayerName
from dg_commons.sim.models.vehicle_ligths import NO_LIGHTS
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from games import UncertaintyParams
from possibilities import PossibilityMonad
from .dg_def import DgSimpleParams, DrivingGame, DrivingGamePlayer
from .preferences_coll_time import VehiclePreferencesCollTime
from .resources_occupancy import ResourcesOccupancy
from .reward_joint import VehicleJointReward
from .reward_personal import VehiclePersonalRewardStructureTime
from .structures import VehicleTrackState
from .vehicle_dynamics import VehicleTrackDynamics
from .visualization import DrivingGameVisualization

__all__ = ["get_driving_game"]


def get_driving_game(dg_params: DgSimpleParams, uncertainty_params: UncertaintyParams) -> DrivingGame:
    ps: PossibilityMonad = uncertainty_params.poss_monad
    players: Dict[PlayerName, DrivingGamePlayer] = {}
    geometries: Dict[PlayerName, VehicleGeometry] = {}
    cc = list(
        cycler(
            color=[
                "c",
                "m",
                "y",
                "b",
                "g",
                "r",
                "gray",
            ]
        )
    )
    resources_occ = ResourcesOccupancy(
        lanelet_network=dg_params.scenario.lanelet_network, cell_resolution=dg_params.shared_resources_ds
    )

    for i, (p, lane) in enumerate(dg_params.ref_lanes.items()):
        g = VehicleGeometry.default_car(
            color=cc[i]["color"],
            w_half=0.8,
            lf=1.6,
            lr=1.6,
        )
        geometries[p] = g

        p_dynamics = VehicleTrackDynamics(
            ref=lane,
            vg=g,
            poss_monad=ps,
            param=dg_params.track_dynamics_param,
            min_safety_distance=D(dg_params.min_safety_distance),
            resources_occupancy=resources_occ,
            goal_progress=dg_params.progress[p][1],
        )
        p_init_progress = dg_params.progress[p][0]
        p_x = VehicleTrackState(
            x=p_init_progress,
            v=dg_params.track_dynamics_param.min_speed + D("2.0"),
            wait=D(0),
            light=NO_LIGHTS,
            has_collided=False,
        )
        p_initial = ps.unit(p_x)
        p_personal_reward_structure = VehiclePersonalRewardStructureTime(
            goal_progress=dg_params.progress[p][1], maximum_depth=dg_params.max_stages
        )
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
        min_safety_distance=dg_params.min_safety_distance,
        players_dynamics={p: players[p].dynamics for p in players},  # temp for quick checking of resources
    )
    game_visualization = DrivingGameVisualization(
        dg_params,
        geometries=geometries,
        plot_limits=dg_params.plot_limits,
        dynamics=fd({p: players[p].dynamics for p in players}),
    )

    game: DrivingGame = DrivingGame(
        players=fd(players),
        ps=ps,
        joint_reward=joint_reward,
        game_visualization=game_visualization,
    )
    return game
