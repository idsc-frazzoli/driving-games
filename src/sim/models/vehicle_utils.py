from sim import logger
from sim.models.vehicle_structures import VehicleParameters


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


def acceleration_constraint(speed: float, acceleration: float, vp: VehicleParameters):
    """Enforces steering limits"""
    if (speed <= vp.vx_limits[0] and acceleration <= 0) or (
            speed >= vp.vx_limits[1] and acceleration >= 0):
        acceleration = 0
        logger.warn("Reached min or max velocity, acceleration set to 0")
    elif acceleration <= vp.acc_limits[0]:
        acceleration = vp.acc_limits[0]
        logger.warn("Commanded acceleration out of limits, clipping value")
    elif acceleration >= vp.acc_limits[1]:
        acceleration = vp.acc_limits[1]
        logger.warn("Commanded acceleration out of limits, clipping value")
    return acceleration
