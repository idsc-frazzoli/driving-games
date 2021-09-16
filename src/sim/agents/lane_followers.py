from typing import Optional

import numpy as np
from duckietown_world.utils import SE2_apply_R2
from geometry import SE2_from_xytheta, SE2value, translation_from_SE2
from dg_commons.controllers.controller_types import *
from dg_commons.controllers.speed import SpeedBehavior, SpeedController
from dg_commons.controllers.steer import SteerController
from dg_commons.planning.lanes import DgLanelet
from dg_commons.planning.trajectory import Trajectory
from sim.agents.lane_follower_z import LFAgent
from sim.models.vehicle import VehicleState
from sim.sim_vis_extra import DrawableTrajectoryType


class LFAgentPP(LFAgent):
    """ This agent is a simple lane follower tracking the centerline of the given lane
    via a pure pursuit controller. The reference in speed is determined by the speed behavior.
    """

    def __init__(self,
                 lane: Optional[DgLanelet] = None,
                 speed_controller: Optional[SpeedController] = None,
                 speed_behavior: Optional[SpeedBehavior] = None,
                 controller: Optional[PurePursuit] = None,
                 steer_controller: Optional[SteeringController] = None,
                 return_extra: bool = False):
        speed_controller: SpeedController = SpeedController() if speed_controller is None else speed_controller
        speed_behavior: SpeedBehavior = SpeedBehavior() if speed_behavior is None else speed_behavior
        steer_controller: SteeringController = SteeringController() if steer_controller is None else steer_controller
        pure_pursuit: PurePursuit = PurePursuit() if controller is None else controller
        super().__init__(lane, pure_pursuit, speed_behavior, speed_controller, steer_controller, return_extra)

    def on_get_extra(self, ) -> Optional[DrawableTrajectoryType]:
        if not self.return_extra:
            return None
        _, gpoint = self.controller.find_goal_point()
        pgoal = translation_from_SE2(gpoint)
        l = self.controller.params.length
        rear_axle = SE2_apply_R2(self.controller.pose, np.array([-l / 2, 0]))
        traj = Trajectory(
            timestamps=[0, 1],
            values=[VehicleState(x=rear_axle[0], y=rear_axle[1], theta=0, vx=0, delta=0),
                    VehicleState(x=pgoal[0], y=pgoal[1], theta=0, vx=1, delta=0),
                    ])
        traj_s = [traj, ]
        colors = ["gold", ]
        return list(zip(traj_s, colors))


class LFAgentFullMPC(LFAgent):
    def __init__(self,
                 lane: Optional[DgLanelet] = None,
                 speed_controller: Optional[SpeedController] = None,
                 speed_behavior: Optional[SpeedBehavior] = None,
                 controller: Optional[LatAndLonController] = None,
                 steer_controller: Optional[SteeringController] = None,
                 return_extra: bool = False):
        speed_controller = None
        speed_behavior: SpeedBehavior = SpeedBehavior() if speed_behavior is None else speed_behavior
        steer_controller: SteeringController = SteeringController() if steer_controller is None else steer_controller
        full_mpc: LatAndLonController = NMPCFullKinContPV() if controller is None else controller
        super().__init__(lane, full_mpc, speed_behavior, speed_controller, steer_controller, return_extra)


class LFAgentLatMPC(LFAgent):
    def __init__(self,
                 lane: Optional[DgLanelet] = None,
                 speed_controller: Optional[SpeedController] = None,
                 speed_behavior: Optional[SpeedBehavior] = None,
                 controller: Optional[LateralController] = None,
                 steer_controller: Optional[SteeringController] = None,
                 return_extra: bool = False):
        speed_controller: SpeedController = SpeedController() if speed_controller is None else speed_controller
        speed_behavior: SpeedBehavior = SpeedBehavior() if speed_behavior is None else speed_behavior
        steer_controller: SteeringController = SteeringController() if steer_controller is None else steer_controller
        lat_mpc: LateralController = NMPCLatKinContPV() if controller is None else controller
        super().__init__(lane, lat_mpc, speed_behavior, speed_controller, steer_controller, return_extra)


class LFAgentStanley(LFAgent):
    def __init__(self,
                 lane: Optional[DgLanelet] = None,
                 speed_controller: Optional[SpeedController] = None,
                 speed_behavior: Optional[SpeedBehavior] = None,
                 controller: Optional[PurePursuit] = None,
                 steer_controller: Optional[SteeringController] = None,
                 return_extra: bool = False):
        speed_controller: SpeedController = SpeedController() if speed_controller is None else speed_controller
        speed_behavior: SpeedBehavior = SpeedBehavior() if speed_behavior is None else speed_behavior
        steer_controller: SteeringController = SteeringController() if steer_controller is None else steer_controller
        stanley: Stanley = Stanley() if controller is None else controller
        super().__init__(lane, stanley, speed_behavior, speed_controller, steer_controller, return_extra)


class LFAgentLQR(LFAgent):
    def __init__(self,
                 lane: Optional[DgLanelet] = None,
                 speed_controller: Optional[SpeedController] = None,
                 speed_behavior: Optional[SpeedBehavior] = None,
                 controller: Optional[LQR] = None,
                 steer_controller: Optional[SteeringController] = None,
                 return_extra: bool = False):
        speed_controller: SpeedController = SpeedController() if speed_controller is None else speed_controller
        speed_behavior: SpeedBehavior = SpeedBehavior() if speed_behavior is None else speed_behavior
        steer_controller: SteeringController = SteeringController() if steer_controller is None else steer_controller
        lqr: LQR = LQR() if controller is None else controller
        super().__init__(lane, lqr, speed_behavior, speed_controller, steer_controller, return_extra)
