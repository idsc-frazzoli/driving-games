__version__ = "1.0.0"

import sys

sys.setrecursionlimit(10000)
from zuper_commons.logs import ZLogger

logger = ZLogger(__name__)

from .structures import *
from .vehicle_dynamics import *
from .vehicle_observation import *
from .preferences_collision import *
from .preferences_coll_time import *
from .collisions import *
from .personal_reward import *
from .joint_reward import *
from .visualization import *
