import math
from dataclasses import dataclass

from sim import logger
from sim.models.model_structures import ModelParameters
from sim.models.utils import rho, kmh2ms


@dataclass(frozen=True, unsafe_hash=True)
class VehicleParameters(ModelParameters):
    delta_max: float
    """ Maximum steering angle [rad] """
    ddelta_max: float
    """ Minimum and Maximum steering rate [rad/s] """

    @classmethod
    def default_car(cls) -> "VehicleParameters":
        # data from https://copradar.com/chapts/references/acceleration.html
        return VehicleParameters(vx_limits=(kmh2ms(-10), kmh2ms(130)),
                                 acc_limits=(-8, 5),
                                 delta_max=math.pi / 6,
                                 ddelta_max=1)

    @classmethod
    def default_bicycle(cls) -> "VehicleParameters":
        return VehicleParameters(vx_limits=(kmh2ms(-1), kmh2ms(50)),
                                 acc_limits=(-4, 3),
                                 delta_max=math.pi / 6,
                                 ddelta_max=1)

    def __post_init__(self):
        super(VehicleParameters, self).__post_init__()
        assert self.delta_max > 0
        assert self.ddelta_max > 0


def steering_constraint(steering_angle: float, steering_velocity: float, vp: VehicleParameters):
    """Enforces steering limits"""
    if (steering_angle <= -vp.delta_max and steering_velocity <= 0) or (
            steering_angle >= vp.delta_max and steering_velocity >= 0):
        steering_velocity = 0
        logger.warn("Reached max steering boundaries")
    elif steering_velocity <= -vp.ddelta_max:
        steering_velocity = -vp.ddelta_max
        logger.warn("Commanded steering rate out of limits, clipping value")
    elif steering_velocity >= vp.ddelta_max:
        steering_velocity = vp.ddelta_max
        logger.warn("Commanded steering rate out of limits, clipping value")
    return steering_velocity


def aerodynamic_force(speed: float, k_drag:float, A: float) -> float:
    """
    :param speed:
    :param A: effective flow surface (frontal area) [m^2]
    :return:
    """
    return .5 * rho * k_drag * A * speed ** 2
