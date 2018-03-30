##############################################################################
# Copyright 2017-2018 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import grove.tomography.operator_utils
from grove.tomography.operator_utils import to_realimag, FROBENIUS, is_projector, EPS, choi_matrix
from grove.tomography.utils import import_qutip, import_cvxpy

qt = import_qutip()
cvxpy = import_cvxpy()

if not qt:
    pytest.skip("Qutip not installed, skipping tests", allow_module_level=True)

if not cvxpy:
    pytest.skip("CVXPY not installed, skipping tests", allow_module_level=True)


def test_operator_basis():
    assert grove.tomography.operator_utils.PAULI_BASIS.all_hermitian()
    assert grove.tomography.operator_utils.PAULI_BASIS.is_orthonormal()
    assert is_projector(grove.tomography.operator_utils.GS)

    two_qubit_pauli = grove.tomography.operator_utils.PAULI_BASIS.product(
        grove.tomography.operator_utils.PAULI_BASIS)
    assert two_qubit_pauli.all_hermitian()
    assert two_qubit_pauli.is_orthonormal()

    sp = grove.tomography.operator_utils.PAULI_BASIS.super_basis()
    assert sp.all_hermitian()
    assert sp.is_orthonormal()

    squared_pauli_basis = grove.tomography.operator_utils.PAULI_BASIS ** 2
    for (l1, o1), (l2, o2) in zip(two_qubit_pauli, squared_pauli_basis):
        assert l1 == l2
        assert (o1 - o2).norm(FROBENIUS) < EPS

    assert np.allclose(
        grove.tomography.operator_utils.PAULI_BASIS.basis_transform.T.toarray() * np.sqrt(2),
        np.array([[1., 0, 0, 1], [0, 1, 1, 0], [0, 1j, -1j, 0], [1, 0, 0, -1]]))

    sX = qt.to_super(grove.tomography.operator_utils.QX)
    tmX = grove.tomography.operator_utils.PAULI_BASIS.transfer_matrix(sX).toarray()
    assert np.allclose(tmX, np.diag([1, 1, -1, -1]))
    assert (sX - grove.tomography.operator_utils.PAULI_BASIS.super_from_tm(tmX)).norm(FROBENIUS) < EPS

    pb3 = grove.tomography.operator_utils.PAULI_BASIS ** 3
    assert pb3.dim == 4**3
    assert pb3 == grove.tomography.operator_utils.n_qubit_pauli_basis(3)

    assert grove.tomography.operator_utils.PAULI_BASIS ** 1 == grove.tomography.operator_utils.PAULI_BASIS

    assert np.allclose(grove.tomography.operator_utils.PAULI_BASIS.project_op(
        grove.tomography.operator_utils.GS).toarray().ravel(),
                       np.array([1, 0, 0, 1]) / np.sqrt(2))

    assert str(grove.tomography.operator_utils.PAULI_BASIS) == "<span[I,X,Y,Z]>"


def test_super_operator_tools():
    X, Y, Z, I = grove.tomography.operator_utils.QX, grove.tomography.operator_utils.QY, grove.tomography.operator_utils.QZ, grove.tomography.operator_utils.QI
    bs = (I, X, Y, Z)

    Xs = qt.sprepost(X, X)
    # verify that Y+XYX==0 ( or XYX==-Y)
    assert (Y + Xs(Y)).norm(FROBENIUS) < EPS

    ptmX = np.array([[(bj * Xs(bk)).tr().real / 2 for bk in bs] for bj in bs])
    assert np.allclose(ptmX, grove.tomography.operator_utils.PAULI_BASIS.transfer_matrix(Xs).toarray())

    xchoi = qt.super_to_choi(Xs)
    my_xchoi = choi_matrix(ptmX, grove.tomography.operator_utils.PAULI_BASIS)
    assert (my_xchoi - xchoi).norm(FROBENIUS) < EPS

    ys = qt.sprepost(Y, Y)
    ptm_y = np.array([[(bj * ys(bk)).tr().real / 2 for bk in bs] for bj in bs])
    assert np.allclose(ptm_y, grove.tomography.operator_utils.PAULI_BASIS.transfer_matrix(ys).toarray())

    ychoi = qt.super_to_choi(ys)
    my_ychoi = choi_matrix(ptm_y, grove.tomography.operator_utils.PAULI_BASIS)
    assert (my_ychoi - ychoi).norm(FROBENIUS) < EPS

    y2 = (-.25j * np.pi * Y).expm()
    y2s = qt.sprepost(y2, y2.dag())
    ptm_y2 = np.array([[(bj * y2s(bk)).tr().real / 2 for bk in bs] for bj in bs])
    assert np.allclose(ptm_y2, grove.tomography.operator_utils.PAULI_BASIS.transfer_matrix(y2s).toarray())

    y2choi = qt.super_to_choi(y2s)
    my_y2choi = choi_matrix(ptm_y2, grove.tomography.operator_utils.PAULI_BASIS)
    assert (my_y2choi - y2choi).norm(FROBENIUS) < EPS


def test_to_realimag():
    op = grove.tomography.operator_utils.QX + grove.tomography.operator_utils.QY
    res = to_realimag(op)
    assert isinstance(res, csr_matrix)
    rd = res.toarray()
    assert np.allclose(rd[:2, :2], [[0, 1], [1, 0]])
    assert np.allclose(rd[:2, 2:], [[0, -1], [1, 0]])
    assert np.allclose(rd[2:, :2], [[0, 1], [-1, 0]])
    assert np.allclose(rd[2:, 2:], [[0, 1], [1, 0]])

    res2 = to_realimag(op.data)
    assert np.allclose(rd, res2.toarray())
