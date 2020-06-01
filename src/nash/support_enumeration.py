from dataclasses import dataclass

from numpy.linalg import lstsq, LinAlgError
import warnings
from itertools import chain, combinations

import numpy as np

__all__ = ["support_enumeration"]


@dataclass(frozen=True, unsafe_hash=True)
class Equilibrium:
    s1: np.ndarray
    s2: np.ndarray
    p1_payoff: float  # fixme
    p2_payoff: float  # fixme


def powerset(n: int):
    """
    A power set of range(n)
    Based on recipe from python itertools documentation:
    https://docs.python.org/2/library/itertools.html#recipes
    """
    return chain.from_iterable(combinations(range(n), r) for r in range(n + 1))


def potential_support_pairs(A, B, non_degenerate=False):
    """
    A generator for the potential support pairs
    Returns
    -------
        A generator of all potential support pairs
    """
    p1_num_strategies, p2_num_strategies = A.shape
    for support1 in (s for s in powerset(p1_num_strategies) if len(s) > 0):
        for support2 in (
            s
            for s in powerset(p2_num_strategies)
            if (len(s) > 0 and not non_degenerate) or len(s) == len(support1)
        ):
            yield support1, support2


def solve_supports(A, B, non_degenerate=False):
    """
    A generator for the strategies corresponding to the potential supports
    Returns
    -------
        A generator of all potential strategies that are indifferent on each
        potential support. Return False if they are not valid (not a
        probability vector OR not fully on the given support).
    """
    for pair in potential_support_pairs(A, B, non_degenerate=non_degenerate):
        s1, s2 = np.zeros(np.shape(A)[0]), np.zeros(np.shape(A)[1])

        if s1.size == 1 and s2.size == 1:
            # if both strategies are singleton no need to solve the linear system
            sub_s1 = np.array([[1]])
            sub_s2 = np.array([[1]])
        else:
            # extract submatrices and solve linear program
            sub_A = A[np.ix_(pair[0], pair[1])]
            sub_B = B[np.ix_(pair[0], pair[1])]
            sub_s1, sub_s2 = solve_linsystem(sub_A, sub_B)

        # rebuild strategies for the full game
        s1[np.ix_(pair[0])] = np.squeeze(sub_s1)
        s2[np.ix_(pair[1])] = np.squeeze(sub_s2)
        yield s1, s2


def _try_lstsq(A, b):
    try:
        x, residuals, _, _ = lstsq(A, b, rcond=None)
        # todo analysis of residuals to check if it is a valid solution?
        return x
    except LinAlgError as e:
        e.args += "Not sure what happened"
        raise e


def solve_linsystem(A: np.ndarray, B: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Let the two strategy vectors be s1 and s2.
    We first solve
        A*s2 = c1 && sum(s2)=1
    then
        s1*B = c2 && sum(s1)=1
    """
    nrow, ncol = A.shape
    zeros_like_s1, zeros_like_s2 = np.zeros([nrow, 1]), np.zeros([1, ncol])
    ones_like_s1, ones_like_s2 = np.ones([nrow, 1]), np.ones([1, ncol])

    # solve system 1
    A_lineq1 = np.concatenate([-ones_like_s1, A], axis=1)
    s2_sum_to_1 = np.concatenate([[[0]], ones_like_s2], axis=1)
    A_lineq1 = np.concatenate([A_lineq1, s2_sum_to_1], axis=0)
    b_lineq1 = np.concatenate([zeros_like_s1, [[1]]], axis=0)
    sol1 = _try_lstsq(A_lineq1, b_lineq1)

    # solve system 2
    A_lineq2 = np.concatenate([-ones_like_s2.T, B.T], axis=1)
    s1_sum_to_1 = np.concatenate([[[0]], ones_like_s1.T], axis=1)
    A_lineq2 = np.concatenate([A_lineq2, s1_sum_to_1], axis=0)
    b_lineq2 = np.concatenate([zeros_like_s2, [[1]]], axis=1)
    sol2 = _try_lstsq(A_lineq2, b_lineq2.T)

    s2 = sol1[1:]
    s1 = sol2[1:]

    return s1, s2


def is_ne(A: np.ndarray, B: np.ndarray, s1: np.ndarray, s2: np.ndarray, tol: float) -> bool:
    """
    Test if a given strategy pair is a pair of best responses
    """
    if (
        np.any(np.logical_or(s1 < 0, 1 < s1))
        or np.any(np.logical_or(s2 < 0, 1 < s2))
        or not np.isclose(sum(s1), 1)
        or not np.isclose(sum(s2), 1)
    ):
        return False
    # Payoff against opponents strategies:
    row_payoffs = np.dot(A, s2)
    column_payoffs = np.dot(s1, B)

    # Payoffs for current candidate strategy
    p1_payoff = np.dot(s1, row_payoffs)
    p2_payoff = np.dot(column_payoffs, s2)
    # must be a best response
    return p1_payoff <= np.min(row_payoffs) + tol and p2_payoff <= np.min(column_payoffs) + tol


def support_enumeration(A: np.ndarray, B: np.ndarray, non_degenerate: bool = False, tol: float = 10 ** -16):
    """
    Obtain the Nash equilibria using support enumeration.
    Algorithm implemented here is Algorithm 3.4 of [Nisan2007]_
    1. For each k in 1...min(size of strategy sets)
    2. For each I,J supports of size k
    3. Solve indifference conditions
    4. Check that have Nash Equilibrium.
    Returns
    -------
        equilibria: A generator.
    """
    count = 0
    for s1, s2 in solve_supports(A, B, non_degenerate=non_degenerate,):
        if is_ne(A, B, s1, s2, tol=tol):
            count += 1
            payoff1 = float(np.dot(np.dot(s1, A), s2))
            payoff2 = float(np.dot(np.dot(s1, B), s2))
            yield Equilibrium(s1, s2, payoff1, payoff2)
    if count % 2 == 0:
        # fixme check why this warning
        warning = """
An even number of ({}) equilibria was returned. This
indicates that the game is degenerate. Consider using another algorithm
to investigate.
                  """.format(
            count
        )
        warnings.warn(warning, RuntimeWarning)
