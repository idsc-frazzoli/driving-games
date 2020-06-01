import numpy as np
from nashpy import support_enumeration as nashpy_sup_enum, vertex_enumeration
from nash import support_enumeration


def test_compare_nashpy():
    """
    The game defined by
        [0, 0, 6, 0, 0]         [3, 0, 2, 1, 0]
    A = [0, 0, 3, 2, 1],    B = [0, 2, 0, 0, 4]
        [4, 3, 0, 0, 1]         [4, 0, 2, 4, 4]
    is supposed to have 8 NE:
y: [ 1 0 0 ]        J1: 0       z: [ 0  1   0   0   0 ]        J2: 0
y: [ 1 0 0 ]        J1: 0       z: [ 0  0   0   0   1 ]        J2: 0
y: [ 2/3 1/3 0 ]    J1: 0       z: [ 0  1   0   0   0 ]        J2: 2/3
y: [ 1/3 1/2 1/6 ]  J1: 4/3     z: [ 0  4/9 2/9 1/3 0 ]        J2: 1
y: [ 0 1/2 1/2 ]    J1: 3/2     z: [ 0  1/2 1/2 0   0 ]        J2: 1
y: [ 0 1 0 ]        J1: 12/7    z: [3/7 0   4/7 0   0 ]        J2: 0
y: [ 0 1 0 ]        J1: 0       z: [1   0   0   0   0 ]        J2: 0
y: [ 0 1 0 ]        J1: 3/2     z: [3/8 0   1/4 3/8 0 ]        J2: 0

    :return:
    """
    A = np.array([[0, 0, 6, 0, 0], [0, 0, 3, 2, 1], [4, 3, 0, 0, 1]])
    B = np.array([[3, 0, 2, 1, 0], [0, 2, 0, 0, 4], [4, 0, 2, 4, 4]])
    for i, eq in enumerate(support_enumeration(A, B)):
        print("My solver: NE ({}): ".format(i + 1), eq)
    for i, eq in enumerate(nashpy_sup_enum(-A, -B, tol=0)):
        print("Nashpy supp enum: NE ({}): ".format(i + 1), eq)
    for i, eq in enumerate(vertex_enumeration(-A, -B)):
        print("Nashpy vertex enum: NE ({}): ".format(i + 1), eq)
