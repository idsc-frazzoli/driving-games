from typing import List
from copy import deepcopy

from reprep import Report
from zuper_commons.text import remove_escapes
from zuper_typing import debug_print

from games.performance import GamePerformance

__all__ = ["report_performance"]


def report_performance(list_game_perf: List[GamePerformance]) -> Report:
    """
    Creates the report for the game performance

    :param list_game_perf: A list containing GamePerformance objects
    :return: The report displaying the game performance info
    """
    r = Report()
    for i, gp in enumerate(list_game_perf):
        _gp = deepcopy(gp)
        st = remove_escapes(debug_print(_gp))
        r.text(f"Run {i}", st)
    return r
