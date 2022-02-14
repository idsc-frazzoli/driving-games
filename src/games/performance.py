from dataclasses import dataclass, field
from typing import Sequence, Optional

from reprep import Report
from zuper_commons.text import remove_escapes


@dataclass
class PerformanceStatistics:
    """Container for collecting some performance statistics. Measures come from perf_counter()."""

    game_name: str
    """The game id. Used to retrieve it from the zoo."""
    solver_name: str
    """The solver id. Used to retrieve it from the zoo."""
    build_joint_game_tree: float = 0
    solve_joint_game_tree: float = 0
    find_factorization: Sequence[float] = field(default_factory=list)
    build_individual_game_trees: Sequence[float] = field(default_factory=list)
    solve_individual_game_trees: Sequence[float] = field(default_factory=list)


def report_performance_stats(perf_stats: PerformanceStatistics) -> Report:
    r = Report(nid=perf_stats.game_name + "-" + perf_stats.solver_name)
    r.text("PerformanceStatistics", remove_escapes(perf_stats.__str__()))
    return r
