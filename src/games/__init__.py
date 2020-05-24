from typing import ClassVar

from zuper_commons.logs import ZLogger

logger = ZLogger(__name__)


class GameConstants:
    checks: ClassVar[bool] = True


from .access import *
from .game_def import *
from .structures_solution import *
from .equilibria import *
from .solution import *
from .reports_player import *
from .reports import *
from .simulate import *
from .single_game_tree import *
from .animations import *
from .zoo import *

#
# # Give each symbol this module name
# for a in list(globals()):
#     v = globals()[a]
#     if hasattr(v, "__module__") and v.__module__.startswith(__name__):
#         logger.info(f"{a} {v.__module__} -> {__name__}")
#         setattr(v, "__module__", __name__)
