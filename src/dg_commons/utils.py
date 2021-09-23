from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import scipy.linalg


@dataclass
class SemiDef:
    eig: Optional[List[float]] = None

    matrix: Optional[np.ndarray] = None

    def __post_init__(self):
        defined_eig: bool = self.eig is not None
        defined_mat: bool = self.matrix is not None

        assert defined_mat or defined_eig

        if defined_eig and type(self.eig) is not list:
            self.eig = self.eig.tolist()

        if defined_eig and defined_mat:
            mat_eig: List[float] = SemiDef.eig_from_semidef(self.matrix)
            mat_eig.sort()
            self.eig.sort()

            assert mat_eig == self.eig
        elif defined_eig:
            self.matrix = SemiDef.semidef_from_eigenvalues(self.eig)
        else:
            self.eig = SemiDef.eig_from_semidef(self.matrix)

        assert all(i >= 0 for i in self.eig)

        assert np.allclose(self.matrix, self.matrix.T)

    @staticmethod
    def semidef_from_eigenvalues(eigenvalues):
        n = len(eigenvalues)
        s = np.diag(eigenvalues)
        q, _ = scipy.linalg.qr(np.random.rand(n, n))
        semidef = q.T @ s @ q
        return semidef

    @staticmethod
    def eig_from_semidef(semidef) -> List[float]:
        eig = np.linalg.eigvalsh(semidef).tolist()
        eig = [round(i, 6) for i in eig]
        return eig

    def __eq__(self, other):
        self.eig.sort()
        other.eig.sort()
        return self.eig == other.eig

    def __lt__(self, other):
        self.eig.sort()
        other.eig.sort()

        return all(self.eig[i] < other.eig[i] for i, _ in enumerate(self.eig))

    def __le__(self, other):
        self.eig.sort()
        other.eig.sort()

        return all(self.eig[i] <= other.eig[i] for i, _ in enumerate(self.eig))
