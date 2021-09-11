from decimal import Decimal

from dg_commons.dynamics import BicycleDynamics
from dg_commons.planning.motion_primitives import MotionPrimitivesGenerator, MPGParam
from sim.models.vehicle_structures import VehicleGeometry
from sim.models.vehicle_utils import VehicleParameters


def test_generate_motion_primitives():
    vp = VehicleParameters.default_car()
    vg = VehicleGeometry.default_car()

    params = MPGParam(dt=Decimal(".2"), n_steps=10, velocity=(0, 50, 5), steering=(-vp.delta_max, vp.delta_max, 7))
    vehicle = BicycleDynamics(vg=vg, vp=vp)
    mpg = MotionPrimitivesGenerator(mpg_param=params,
                                    vehicle_dynamics=vehicle.successor,
                                    vehicle_params=VehicleParameters.default_car())

    traject = mpg.generate_motion_primitives()
    print(traject)


