from typing import Union
from dg_commons.controllers.pure_pursuit_z import PurePursuit, PurePursuitParam
from dg_commons.controllers.stanley_controller import Stanley, StanleyParam
from dg_commons.controllers.speed import SpeedController, SpeedBehavior, SpeedControllerParam, SpeedBehaviorParam
from dg_commons.controllers.steering_controllers import SCP, SCIdentity, SCIdentityParam, SCPParam
from dg_commons.controllers.lqr import LQR, LQRParam
from dg_commons.controllers.mpc.nmpc_lateral_kin_cont import NMPCLatKinCont, NMPCLatKinContParam
from dg_commons.controllers.mpc.nmpc_lateral_kin_dis import NMPCLatKinDis, NMPCLatKinDisParam
from dg_commons.controllers.mpc.nmpc_full_kin_cont import NMPCFullKinCont, NMPCFullKinContParam
from dg_commons.controllers.mpc.nmpc_full_kin_dis import NMPCFullKinDis, NMPCFullKinDisParam


class Empty:
    pass


LateralController = Union[Empty, PurePursuit, Stanley, LQR, NMPCLatKinCont, NMPCLatKinCont, NMPCLatKinDis]
LateralControllerParam = Union[Empty, PurePursuitParam, StanleyParam, LQRParam, NMPCLatKinContParam, NMPCLatKinDisParam]

LongitudinalController = Union[Empty, SpeedController]
LongitudinalControllerParam = Union[Empty, SpeedControllerParam]

LongitudinalBehavior = Union[Empty, SpeedBehavior]
LongitudinalBehaviorParam = Union[Empty, SpeedBehaviorParam]

SteeringController = Union[Empty, SCP, SCIdentity]
SteeringControllerParam = Union[Empty, SCPParam, SCIdentityParam]

LatAndLonController = Union[Empty, NMPCFullKinDis, NMPCFullKinCont]
LatAndLonControllerParam = Union[Empty, NMPCFullKinContParam, NMPCFullKinDisParam]
