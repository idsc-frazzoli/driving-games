import sys

sys.setrecursionlimit(10000)
from zuper_commons.logs import ZLogger

logger = ZLogger(__name__)

from .game_generation import *
from .zoo import *
