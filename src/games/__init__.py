import logging
from typing import ClassVar

from zuper_commons.logs import ZLogger

logger = ZLogger(__name__)
logging.getLogger("matplotlib.font_manager").disabled = True


class GameConstants:
    """Global constants for the program."""

    checks: ClassVar[bool] = False  # True  # False
    """
        If true activates extensive checks and assertions.
        Slows down the solving a lot.
    """


from .preprocess import *
from .game_def import *
from games.solve.solution_structures import *
from games.solve.equilibria import *
from games.solve.solution import *
from .reports_player import *
from .reports import *
from .simulate import *
from .report_animations import *
