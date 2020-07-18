__version__ = "1.0.0"

import sys

sys.setrecursionlimit(10000)
from zuper_commons.logs import ZLogger

logger = ZLogger(__name__)

from .game_generation import *
from driving_games.structures import *
from driving_games.visualization import *
from driving_games.preferences_collision import *
from driving_games.preferences_coll_time import *
from driving_games.joint_reward import *
from driving_games.vehicle_observation import *
from driving_games.personal_reward import *
from .zoo import *
from driving_games.collisions import *