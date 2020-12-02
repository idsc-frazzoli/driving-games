__version__ = "1.0.0"

import sys

sys.setrecursionlimit(10000)
from zuper_commons.logs import ZLogger

logger = ZLogger(__name__)

from .sequence import *
from .transitions import *
from .structures import *
from .world import *
from .rules import *
from .metrics import *
from .bicycle_dynamics import *
from .trajectory_graph import *
