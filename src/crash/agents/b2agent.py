from decimal import Decimal
from itertools import product
from typing import Optional

from dg_commons.dynamics import BicycleDynamics
from dg_commons.planning.motion_primitives import MPGParam, MotionPrimitivesGenerator
from dg_commons.sim.agents.lane_follower import LFAgent
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim import DrawableTrajectoryType

__all__ = ["B2Agent"]


class B2Agent(LFAgent):
    """Baseline 2 agent. In case of emergency generate escape trajectories"""

    def __init__(self, *args, **kwargs):
        super(B2Agent, self).__init__(*args, **kwargs)
        # todo init from vehicle parameters
        vp = VehicleParameters.default_car()
        vg = VehicleGeometry.default_car()
        mpg_param = MPGParam(
            dt=Decimal(".2"), n_steps=10, velocity=(0, 10, 5), steering=(-vp.delta_max, vp.delta_max, 9)
        )
        vehicle = BicycleDynamics(vg=vg, vp=vp)
        self._mpg = MotionPrimitivesGenerator(param=mpg_param, vehicle_dynamics=vehicle.successor, vehicle_param=vp)

    def on_get_extra(
        self,
    ) -> Optional[DrawableTrajectoryType]:
        trajectories = self._mpg.generate(x0=self._my_obs.to_vehicle_state())
        if len(trajectories) == 0:
            return None
        candidates = tuple(
            product(
                trajectories,
                [
                    "gold",
                ],
            )
        )
        return candidates
