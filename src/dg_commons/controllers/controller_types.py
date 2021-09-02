from typing import Union
from src.dg_commons.controllers.pure_pursuit import PurePursuit, PurePursuitParam
from src.dg_commons.controllers.stanley_controller import Stanley, StanleyParam
from src.dg_commons.controllers.speed import SpeedController, SpeedBehavior, SpeedControllerParam, SpeedBehaviorParam
from src.dg_commons.controllers.steering_controllers import SCP, SCIdentity, SCIdentityParam, SCPParam

LateralController = Union[PurePursuit, Stanley]
LateralControllerParam = Union[PurePursuitParam, StanleyParam]
LongitudinalController = Union[SpeedController]
LongitudinalControllerParam = Union[SpeedControllerParam]
LongitudinalBehavior = Union[SpeedBehavior]
LongitudinalBehaviorParam = Union[SpeedBehaviorParam]
SteeringController = Union[SCP, SCIdentity]
SteeringControllerParam = Union[SCPParam, SCIdentityParam]
