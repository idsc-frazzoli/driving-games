from dataclasses import dataclass, field
from typing import Sequence, Optional


@dataclass
class PerformanceStatistics:
    """Container for collecting some performance statistics. Measures come from perf_counter()."""

    game_id: str
    """The game id. Used to retrieve it from the zoo."""
    solver_id: str
    """The solver id. Used to retrieve it from the zoo."""
    build_joint_game_tree: float
    solve_joint_game_tree: float
    find_factorization: Optional[float] = None
    build_individual_game_trees: Sequence[float] = field(default_factory=list)
    solve_individual_game_trees: Sequence[float] = field(default_factory=list)
