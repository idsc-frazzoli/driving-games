__version__ = "1.0.0"

import sys

sys.setrecursionlimit(10000)
from zuper_commons.logs import ZLogger

logger = ZLogger(__name__)

from .structures import *
from .vehicle_dynamics import *
from .vehicle_observation import *
from .collisions import *
from .collisions_check import *
from .preferences_coll_time import *
from .preferences_collision import *
from .reward_personal import *
from .reward_joint import *
from .visualization import *
#from .demo import *
