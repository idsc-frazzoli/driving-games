from typing import List
from copy import deepcopy

from reprep import Report
from zuper_commons.text import remove_escapes
from zuper_typing import debug_print

from games.performance import GamePerformance

__all__ = ["report_performance"]


def report_performance(list_game_perf: List[GamePerformance]):
    r = Report()
    for i, gp in enumerate(list_game_perf):
        _gp = deepcopy(gp)
        _gp.pre_pro_player_pi.get_fact_pi.find_dependencies_times = "Too long to display"
        st = remove_escapes(debug_print(_gp))
        r.text(f"Run {i}", st)
    return r
