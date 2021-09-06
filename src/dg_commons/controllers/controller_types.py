from typing import Union
from dg_commons.controllers.pure_pursuit import PurePursuit, PurePursuitParam
from dg_commons.controllers.stanley_controller import Stanley, StanleyParam
from dg_commons.controllers.speed import SpeedController, SpeedBehavior, SpeedControllerParam, SpeedBehaviorParam
from dg_commons.controllers.steering_controllers import SCP, SCIdentity, SCIdentityParam, SCPParam


class Empty:
    pass


LateralController = Union[Empty, PurePursuit, Stanley]
LateralControllerParam = Union[Empty, PurePursuitParam, StanleyParam]

LongitudinalController = Union[Empty, SpeedController]
LongitudinalControllerParam = Union[Empty, SpeedControllerParam]

LongitudinalBehavior = Union[Empty, SpeedBehavior]
LongitudinalBehaviorParam = Union[Empty, SpeedBehaviorParam]

SteeringController = Union[Empty, SCP, SCIdentity]
SteeringControllerParam = Union[Empty, SCPParam, SCIdentityParam]

LatAndLonController = Union[Empty]
LatAndLonControllerParam = Union[Empty]
