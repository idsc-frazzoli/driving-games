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
