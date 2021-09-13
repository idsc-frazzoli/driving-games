from typing import Union
from dg_commons.controllers.pure_pursuit import PurePursuit, PurePursuitParam
from dg_commons.controllers.stanley_controller import Stanley, StanleyParam
from dg_commons.controllers.speed import SpeedController, SpeedBehavior, SpeedControllerParam, SpeedBehaviorParam
from dg_commons.controllers.steering_controllers import SCP, SCIdentity, SCIdentityParam, SCPParam
from dg_commons.controllers.lqr import LQR, LQRParam
from dg_commons.controllers.mpc.mpc_kin_cont import MPCKinCont, MPCKinContParam
from dg_commons.controllers.mpc.mpc_kin_cont_path_var import MPCKinContPathVarParam, MPCKinContPathVar
from dg_commons.controllers.mpc.mpc_kin_dis import MPCKinDisParam, MPCKinDis
from dg_commons.controllers.mpc.mpc_kin_full_cont import MPCKinContFullParam, MPCKinContFull
from dg_commons.controllers.mpc.mpc_kin_full_dis import MPCKinDisFullParam, MPCKinDisFull


class Empty:
    pass


LateralController = Union[Empty, PurePursuit, Stanley, LQR, MPCKinCont, MPCKinContPathVar, MPCKinDis]
LateralControllerParam = Union[Empty, PurePursuitParam, StanleyParam, LQRParam, MPCKinContParam, MPCKinContPathVarParam,
                               MPCKinDisParam]

LongitudinalController = Union[Empty, SpeedController]
LongitudinalControllerParam = Union[Empty, SpeedControllerParam]

LongitudinalBehavior = Union[Empty, SpeedBehavior]
LongitudinalBehaviorParam = Union[Empty, SpeedBehaviorParam]

SteeringController = Union[Empty, SCP, SCIdentity]
SteeringControllerParam = Union[Empty, SCPParam, SCIdentityParam]

LatAndLonController = Union[Empty, MPCKinContFull, MPCKinDisFull]
LatAndLonControllerParam = Union[Empty, MPCKinContFullParam, MPCKinDisFullParam]
