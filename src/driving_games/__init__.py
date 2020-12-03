__version__ = "1.0.0"

import sys

sys.setrecursionlimit(10000)
from zuper_commons.logs import ZLogger

logger = ZLogger(__name__)

from .game_generation import *
from .structures import *
from .visualization import *
from .preferences_collision import *
from .preferences_coll_time import *
from .joint_reward import *
from .vehicle_observation import *
from .personal_reward import *
from .zoo import *
from .collisions import *
from .rectangle import *
