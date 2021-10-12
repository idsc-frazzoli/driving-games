import numpy as np
from casadi import *
from dg_commons.utils import SemiDef
from dataclasses import dataclass
from typing import Union, List, Dict
from dg_commons.utils import BaseParams


class Empty:
    pass


@dataclass
class QuadraticParams(BaseParams):
    q: Union[List[SemiDef], SemiDef] = SemiDef([0])
    r: Union[List[SemiDef], SemiDef] = SemiDef([0])


@dataclass
class TestParams(BaseParams):
    a: float = 1


def quadratic_cost(x, u, quad_params):
    r = SX(quad_params.r.matrix)
    q = SX(quad_params.q.matrix)

    dim_x = len(x)
    dim_u = len(u)
    helper1 = GenSX_zeros(dim_x)
    helper2 = GenSX_zeros(dim_u)

    for i in range(dim_x):
        helper1[i] = x[i]

    for i in range(dim_u):
        helper2[i] = u[i]

    return bilin(q, helper1, helper1) + bilin(r, helper2, helper2), bilin(q, helper1, helper1)

def f():
    return


CostParameters = Union[Empty, QuadraticParams, TestParams]
costs = {"quadratic": quadratic_cost, "Test": f}
MapCostParam: Dict[str, type(CostParameters)] = {"quadratic": QuadraticParams, "Test": TestParams}

assert set(costs.keys()) == set(MapCostParam.keys())
