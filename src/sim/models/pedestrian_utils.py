import math
from dataclasses import dataclass

from sim import logger
from sim.models.model_structures import ModelParameters
from sim.models.utils import kmh2ms


@dataclass(frozen=True, unsafe_hash=True)
class PedestrianParameters(ModelParameters):
    dtheta_max: float
    """ Maximum turning_rate [rad] """

    @classmethod
    def default(cls) -> "PedestrianParameters":
        # data from https://copradar.com/chapts/references/acceleration.html
        return PedestrianParameters(vx_limits=(kmh2ms(-5), kmh2ms(15)),
                                    acc_limits=(-3, 3),
                                    dtheta_max=math.pi
                                    )

    def __post_init__(self):
        super(PedestrianParameters, self).__post_init__()
        assert self.dtheta_max > 0


def rotation_constraint(rot_velocity: float, pp: PedestrianParameters):
    """Enforces rotation limits"""

    if rot_velocity <= -pp.dtheta_max:
        rot_velocity = -pp.dtheta_max
        logger.warn("Commanded pedestrian turning rate out of limits, clipping value")
    elif rot_velocity >= pp.dtheta_max:
        rot_velocity = pp.dtheta_max
        logger.warn("Commanded pedestrian turning rate out of limits, clipping value")
    return rot_velocity
