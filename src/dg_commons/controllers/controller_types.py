from typing import Union
from src.dg_commons.controllers.pure_pursuit import PurePursuit
from src.dg_commons.controllers.speed import SpeedController, SpeedBehavior
from src.dg_commons.controllers.steering_controllers import SCP, SCIdentityParam

LateralController = Union[PurePursuit]
LongitudinalController = Union[SpeedController]
LongitudinalBehavior = Union[SpeedBehavior]
SteeringController = Union[SCP, SCIdentityParam]
