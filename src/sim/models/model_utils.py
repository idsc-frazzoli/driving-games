from sim import logger
from sim.models.model_structures import ModelParameters


def acceleration_constraint(speed: float, acceleration: float, p: ModelParameters):
    """Enforces acceleration limits"""
    if (speed <= p.vx_limits[0] and acceleration <= 0) or (
            speed >= p.vx_limits[1] and acceleration >= 0):
        acceleration = 0
        logger.warn("Reached min or max velocity, acceleration set to 0")
    elif acceleration <= p.acc_limits[0]:
        acceleration = p.acc_limits[0]
        logger.warn("Commanded acceleration out of limits, clipping value")
    elif acceleration >= p.acc_limits[1]:
        acceleration = p.acc_limits[1]
        logger.warn("Commanded acceleration out of limits, clipping value")
    return acceleration
