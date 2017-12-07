import pytest
from mock import Mock, patch

from pyquil.gates import X
from pyquil.quil import Program
import numpy as np
from scipy.sparse import csr_matrix
import qutip as qt

import grove.tomography.utils as ut
import grove


def test_operator_basis():
    assert ut.PAULI_BASIS.all_hermitian()
    assert ut.PAULI_BASIS.is_orthonormal()
    assert ut.is_projector(ut.GS)

    p2 = ut.PAULI_BASIS.product(ut.PAULI_BASIS)
    assert p2.all_hermitian()
    assert p2.is_orthonormal()

    sp = ut.PAULI_BASIS.super_basis()
    assert sp.all_hermitian()
    assert sp.is_orthonormal()

    p2_2 = ut.PAULI_BASIS ** 2
    for (l1, o1), (l2, o2) in zip(p2, p2_2):
        assert l1 == l2
        assert (o1 - o2).norm('fro') < ut.EPS

    assert np.allclose(ut.PAULI_BASIS.basis_transform.T.toarray() * np.sqrt(2),
                       np.array([[1., 0, 0, 1], [0, 1, 1, 0], [0, 1j, -1j, 0], [1, 0, 0, -1]]))

    sX = qt.to_super(ut.qX)
    tmX = ut.PAULI_BASIS.transfer_matrix(sX).toarray()
    assert np.allclose(tmX, np.diag([1,1,-1,-1.]))
    assert (sX - ut.PAULI_BASIS.super_from_tm(tmX)).norm('fro') < ut.EPS

    pb3 = ut.PAULI_BASIS**3
    assert pb3.dim == 4**3
    assert pb3 == ut.n_qubit_pauli_basis(3)

    assert ut.PAULI_BASIS**1 == ut.PAULI_BASIS

    assert np.allclose(ut.PAULI_BASIS.project_op(ut.GS).toarray().ravel(),
                       np.array([1,0,0,1])/np.sqrt(2))

    assert str(ut.PAULI_BASIS) == "<span[I,X,Y,Z]>"

    gmb = ut.gell_mann_basis(2)
    assert np.allclose((gmb.basis_transform.H * ut.PAULI_BASIS.basis_transform).toarray(),
                       np.eye(4))

    gmb3 = ut.gell_mann_basis(3)
    assert gmb3.is_orthonormal()
    assert gmb3.all_hermitian()
    assert gmb3.dim == 9

    gmb4 = ut.gell_mann_basis(4)
    assert gmb4.is_orthonormal()
    assert gmb4.all_hermitian()
    assert gmb4.dim == 16


def test_sample_outcomes_make_histogram():
    n = 10
    N = 10000

    np.random.seed(2342)
    probs = np.random.rand(n)
    probs /= probs.sum()

    histogram = ut.make_histogram(ut.sample_outcomes(probs, N), n) / float(N)
    assert np.allclose(histogram, probs, atol=.01)


def test_basis_state_preps():
    II, IX, XI, XX = ut.basis_state_preps(0, 1)
    assert II.out() == "I 0\nI 1\n"
    assert IX.out() == "I 0\nX 1\n"
    assert XI.out() == "X 0\nI 1\n"
    assert XX.out() == "X 0\nX 1\n"


def test_sample_bad_readout():
    np.random.seed(234)
    assignment_probs = .3*np.random.rand(4, 4) + np.eye(4)
    assignment_probs /= assignment_probs.sum(axis=0)[np.newaxis, :]
    cxn = Mock()
    cxn.wavefunction.return_value.amplitudes = 0.5j * np.ones(4)
    with patch("grove.tomography.utils.sample_outcomes") as so:
        ut.sample_bad_readout(Program(X(0), X(1), X(0), X(1)), 10000, assignment_probs, cxn)
        assert np.allclose(so.call_args[0][0], 0.25 * assignment_probs.sum(axis=1))


def test_estimate_assignment_probs():
    np.random.seed(2345)
    outcomes = np.random.randint(0, 1000, size=(4, 4))
    aprobs = ut.estimate_assignment_probs(outcomes)
    assert np.allclose(aprobs.T * outcomes.sum(axis=1)[:, np.newaxis], outcomes)


def test_visualization():
    ax = Mock()
    with patch("grove.tomography.utils.matrix_histogram_complex",
               spec=grove.tomography.utils.matrix_histogram_complex) as mhc:
        ut.state_histogram(ut.GS, ax, "test")
    assert mhc.called
    assert ax.view_init.called
    assert ax.set_title.called

    ptX = ut.PAULI_BASIS.transfer_matrix(qt.to_super(ut.qX)).toarray()
    ax = Mock()
    with patch("matplotlib.pyplot.colorbar"):
        ut.plot_pauli_transfer_matrix(ptX, ax, ut.PAULI_BASIS.labels, "bla")
    assert ax.imshow.called
    assert ax.set_xlabel.called
    assert ax.set_ylabel.called


def test_product_basis():
    X, Y, Z, I = ut.qX, ut.qY, ut.qZ, ut.qI

    assert ut.is_hermitian(X.data.toarray())

    labels = ["II", "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX",
        "ZY", "ZZ"]
    d = {"I": I / np.sqrt(2), "X": X / np.sqrt(2), "Y": Y / np.sqrt(2), "Z": Z / np.sqrt(2)}
    ops = [qt.tensor(d[s[0]], d[s[1]]) for s in labels]

    for ((ll1, op1), ll2, op2) in zip(ut.PAULI_BASIS.product(ut.PAULI_BASIS), labels, ops):
        assert ll1 == ll2
        assert (op1 - op2).norm('fro') < ut.EPS


def test_super_operator_tools():
    X, Y, Z, I = ut.qX, ut.qY, ut.qZ, ut.qI
    bs = (I, X, Y, Z)

    Xs = qt.sprepost(X, X)
    # verify that Y+XYX==0 ( or XYX==-Y)
    assert (Y + Xs(Y)).norm('fro') < ut.EPS

    ptmX = np.array([[(bj * Xs(bk)).tr().real / 2 for bk in bs] for bj in bs])
    assert np.allclose(ptmX, ut.PAULI_BASIS.transfer_matrix(Xs).toarray())

    Xchoi = qt.super_to_choi(Xs)
    myXchoi = ut.choi_matrix(ptmX, ut.PAULI_BASIS)
    assert (myXchoi - Xchoi).norm('fro') < ut.EPS

    Ys = qt.sprepost(Y, Y)
    ptmY = np.array([[(bj * Ys(bk)).tr().real / 2 for bk in bs] for bj in bs])
    assert np.allclose(ptmY, ut.PAULI_BASIS.transfer_matrix(Ys).toarray())

    Ychoi = qt.super_to_choi(Ys)
    myYchoi = ut.choi_matrix(ptmY, ut.PAULI_BASIS)
    assert (myYchoi - Ychoi).norm('fro') < ut.EPS

    Y2 = (-.25j * np.pi * Y).expm()
    Y2s = qt.sprepost(Y2, Y2.dag())
    ptmY2 = np.array([[(bj * Y2s(bk)).tr().real / 2 for bk in bs] for bj in bs])
    assert np.allclose(ptmY2, ut.PAULI_BASIS.transfer_matrix(Y2s).toarray())

    Y2choi = qt.super_to_choi(Y2s)
    myY2choi = ut.choi_matrix(ptmY2, ut.PAULI_BASIS)
    assert (myY2choi - Y2choi).norm('fro') < ut.EPS


def test_states():
    preparations = ut.qX, ut.qY, ut.qZ
    states = ut.generated_states(ut.GS, preparations)
    assert (states[0] - ut.ES).norm('fro') < ut.EPS
    assert (states[1] - ut.ES).norm('fro') < ut.EPS
    assert (states[2] - ut.GS).norm('fro') < ut.EPS
    assert (ut.n_qubit_ground_state(2) - qt.tensor(ut.GS, ut.GS)).norm('fro') < ut.EPS


def test_matrix_props():
    assert ut.is_hermitian(ut.qX)
    assert ut.is_hermitian(ut.qX.data)
    assert ut.is_hermitian(ut.qX.data.toarray())

    assert ut.is_projector(ut.GS)


def test_to_realimag():
    op = ut.qX + ut.qY
    res = ut.to_realimag(op)
    assert isinstance(res, csr_matrix)
    rd = res.toarray()
    assert np.allclose(rd[:2, :2], [[0, 1], [1, 0]])
    assert np.allclose(rd[:2, 2:], [[0, -1], [1, 0]])
    assert np.allclose(rd[2:, :2], [[0, 1], [-1, 0]])
    assert np.allclose(rd[2:, 2:], [[0, 1], [1, 0]])

    res2 = ut.to_realimag(op.data)
    assert np.allclose(rd, res2.toarray())


def test_povm():
    pi_basis = ut.POVM_PI_BASIS
    confusion_rate_matrix = np.eye(2)
    povm = ut.make_diagonal_povm(pi_basis, confusion_rate_matrix)

    assert (povm.ops[0] - pi_basis.ops[0]).norm('fro') < ut.EPS
    assert (povm.ops[1] - pi_basis.ops[1]).norm('fro') < ut.EPS

    with pytest.raises(ut.CRMUnnormalizedError):
        confusion_rate_matrix = np.array([[.8, 0.],
                                     [.3, 1.]])
        povm = ut.make_diagonal_povm(pi_basis, confusion_rate_matrix)

    with pytest.raises(ut.CRMValueError):
        confusion_rate_matrix = np.array([[.8, -.1],
                                     [.2, 1.1]])
        povm = ut.make_diagonal_povm(pi_basis, confusion_rate_matrix)


def test_basis_labels():
    for num_qubits, desired_labels in [(1, ['0', '1']),
                                       (2, ['00', '01', '10', '11'])]:
        generated_labels = ut.basis_labels(num_qubits)
        assert desired_labels == generated_labels
