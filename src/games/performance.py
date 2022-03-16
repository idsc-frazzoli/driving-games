from dataclasses import dataclass, field
from typing import List

from reprep import Report
from zuper_commons.text import remove_escapes
from zuper_typing import debug_print


@dataclass
class PerformanceStatistics:
    """Container for collecting some performance statistics. Measures come from perf_counter()."""

    game_name: str
    """The game id. Used to retrieve it from the zoo."""
    solver_name: str
    """The solver id. Used to retrieve it from the zoo."""
    build_joint_game_tree: float = 0
    """Time to build the joint game tree."""
    joint_game_tree_nodes: int = 0
    """Number of nodes in the joint game tree."""
    solve_joint_game_graph: float = 0
    """Time to solve the joint game graph."""
    factorization_time: float = field(default_factory=list)
    """Time spent factorizing the game"""
    build_individual_game_graphs: List[float] = field(default_factory=list)
    """Time to build the individual game graphs."""
    solve_individual_game_graphs: List[float] = field(default_factory=list)
    """Time to solve the individual game graphs."""
    individual_game_graphs_nodes: List[int] = field(default_factory=list)
    """Number of nodes in the individual game graphs."""


def report_performance_stats(perf_stats: PerformanceStatistics) -> Report:
    r = Report(nid="performance-stats")
    r.text("PerformanceStatistics", remove_escapes(debug_print(perf_stats)))
    total_time_ind = sum(perf_stats.build_individual_game_graphs) + sum(perf_stats.solve_individual_game_graphs)
    total_time_joint = perf_stats.build_joint_game_tree + perf_stats.solve_joint_game_graph
    total_individual_gg_nodes = sum(perf_stats.individual_game_graphs_nodes)
    times_str = (
        f"Total time ind:\t{total_time_ind:.2f}\n"
        + f"Total time joint:\t{total_time_joint:.2f}\n"
        + f"Total time:\t{total_time_joint + total_time_ind:.2f}\n"
        + f"Individual game graph nodes:\t{total_individual_gg_nodes}\n"
        + f"Total game nodes:\t{total_individual_gg_nodes + perf_stats.joint_game_tree_nodes}"
    )
    r.text("PerformanceStatistics2", remove_escapes(debug_print(times_str)))

    return r
