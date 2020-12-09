__version__ = "1.0.0"

import sys

sys.setrecursionlimit(10000)
from zuper_commons.logs import ZLogger

logger = ZLogger(__name__)

from .sequence import *
from .paths import *
from .structures import *
from .world import *
from .metrics_def import *
from .metrics import *
from .bicycle_dynamics import *
from .trajectory_graph import *
from .trajectory_generator import *
from .trajectory_game import *
