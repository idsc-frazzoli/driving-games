from casadi import *
from typing import Union
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import scipy.linalg


@dataclass
class SemiDef:
    """
    Positive Semi-definite Matrices with Ellipse Inclusion Order

    The order induced by eq, lt le is as follows:
    Semidef1 <= Semidef2 if sort(eig(Semidef1))_i <= sort(eig(Semidef2))_i for all i meaning that the ellipse induced
    by the quadratic form of Semidef1 is included or equal to the ellipse induced by the quadratic form of Semidef2.
    """

    eig: List[float] = None
    """ List of eigenvalues """

    matrix: np.ndarray = None
    """ Matrix as n x n numpy array """

    def __post_init__(self) -> None:
        """
        Checks that either the eigenvalues or the matrix itself is defined.
        If both of them are passed, it is checked that they match.
        """
        defined_eig: bool = self.eig is not None
        defined_mat: bool = self.matrix is not None

        assert defined_mat or defined_eig

        if defined_eig and type(self.eig) is not list:
            self.eig: List[float] = self.eig.tolist()

        if defined_eig and defined_mat:
            mat_eig: List[float] = SemiDef.eig_from_matrix(self.matrix)
            mat_eig.sort()
            self.eig.sort()

            assert mat_eig == self.eig
        elif defined_eig:
            self.matrix: np.ndarray = SemiDef.matrix_from_eigenvalues(self.eig)
        else:
            self.eig: List[float] = SemiDef.eig_from_matrix(self.matrix)

        assert all(i >= 0 for i in self.eig)

        assert np.allclose(self.matrix, self.matrix.T)

    @staticmethod
    def matrix_from_eigenvalues(eigenvalues: List[float]) -> np.ndarray:
        """
        Creates a random matrix with the eigenvalues passed as arguments.
        @param eigenvalues: The desired eigenvalues.
        @return: The computed matrix.
        """
        n: int = len(eigenvalues)
        s: np.ndarray = np.diag(eigenvalues)
        q, _ = scipy.linalg.qr(np.random.rand(n, n))
        semidef: np.ndarray = q.T @ s @ q
        return semidef

    @staticmethod
    def eig_from_matrix(semidef: np.ndarray) -> List[float]:
        """
        Computes the eigenvalues
        @param semidef: nxn matrix.
        @return: its eigenvalues
        """
        eig = np.linalg.eigvalsh(semidef).tolist()
        eig: List[float] = [round(i, 6) for i in eig]
        return eig

    def __eq__(self, other):
        return self.matrix == other.matrix

    def __lt__(self, other):
        self.eig.sort()
        other.eig.sort()

        return all(self.eig[i] < other.eig[i] for i, _ in enumerate(self.eig))

    def __le__(self, other):
        self.eig.sort()
        other.eig.sort()

        return all(self.eig[i] <= other.eig[i] for i, _ in enumerate(self.eig))


@dataclass
class QuadraticParams:
    q: Union[List[SemiDef], SemiDef] = SemiDef([0])
    r: Union[List[SemiDef], SemiDef] = SemiDef([0])


class QuadraticCost:
    def __init__(self, params: QuadraticParams):
        self.params = params

    def cost_function(self, x, u):
        r = SX(self.params.r.matrix)
        q = SX(self.params.q.matrix)

        dim_x = len(x)
        dim_u = len(u)
        helper1 = GenSX_zeros(dim_x)
        helper2 = GenSX_zeros(dim_u)

        for i in range(dim_x):
            helper1[i] = x[i]

        for i in range(dim_u):
            helper2[i] = u[i]

        return bilin(q, helper1, helper1) + bilin(r, helper2, helper2), bilin(q, helper1, helper1)


CostFunctions = Union[QuadraticCost]
CostParameters = Union[QuadraticParams]
MapCostParam: Dict[type(CostFunctions), type(CostParameters)] = {QuadraticCost: QuadraticParams}
