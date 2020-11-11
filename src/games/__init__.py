from typing import ClassVar

from zuper_commons.logs import ZLogger

logger = ZLogger(__name__)


class GameConstants:
    """ Global constants for the program. """

    checks: ClassVar[bool] = False
    """ 
        If true activates extensive checks and assertions. 
        Slows down the solving a lot. 
    """


from .access import *
from .game_def import *
from .structures_solution import *
from .equilibria import *
from .solution import *
from .reports_player import *
from .reports import *
from .simulate import *

from .report_animations import *
from .zoo import *
