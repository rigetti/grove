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
from matplotlib.pyplot import figure
from unittest.mock import Mock, patch, call
from mpl_toolkits.mplot3d import Axes3D
from pyquil.api import QuantumComputer
from pyquil.gates import X, Y, I
from pyquil.quil import Program

import grove.tomography.operator_utils
import grove.tomography.operator_utils as o_ut
import grove.tomography.utils as ut
from grove.tomography.operator_utils import FROBENIUS

qt = ut.import_qutip()
cvxpy = ut.import_cvxpy()

if not qt:
    pytest.skip("Qutip not installed, skipping tests", allow_module_level=True)

if not cvxpy:
    pytest.skip("CVXPY not installed, skipping tests", allow_module_level=True)


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
    assert II.out() == "PRAGMA PRESERVE_BLOCK\nI 0\nI 1\nPRAGMA END_PRESERVE_BLOCK\n"
    assert IX.out() == "PRAGMA PRESERVE_BLOCK\nI 0\nX 1\nPRAGMA END_PRESERVE_BLOCK\n"
    assert XI.out() == "PRAGMA PRESERVE_BLOCK\nX 0\nI 1\nPRAGMA END_PRESERVE_BLOCK\n"
    assert XX.out() == "PRAGMA PRESERVE_BLOCK\nX 0\nX 1\nPRAGMA END_PRESERVE_BLOCK\n"


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


def test_product_basis():
    X, Y, Z, I = grove.tomography.operator_utils.QX, grove.tomography.operator_utils.QY, grove.tomography.operator_utils.QZ, grove.tomography.operator_utils.QI

    assert o_ut.is_hermitian(X.data.toarray())

    labels = ["II", "IX", "IY", "IZ", "XI", "XX", "XY", "XZ",
              "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]
    d = {"I": I / np.sqrt(2), "X": X / np.sqrt(2), "Y": Y / np.sqrt(2), "Z": Z / np.sqrt(2)}
    ops = [qt.tensor(d[s[0]], d[s[1]]) for s in labels]

    for ((ll1, op1), ll2, op2) in zip(grove.tomography.operator_utils.PAULI_BASIS.product(
            grove.tomography.operator_utils.PAULI_BASIS), labels, ops):
        assert ll1 == ll2
        assert (op1 - op2).norm(FROBENIUS) < o_ut.EPS


def test_states():
    preparations = grove.tomography.operator_utils.QX, grove.tomography.operator_utils.QY, grove.tomography.operator_utils.QZ
    states = ut.generated_states(grove.tomography.operator_utils.GS, preparations)
    assert (states[0] - grove.tomography.operator_utils.ES).norm(FROBENIUS) < o_ut.EPS
    assert (states[1] - grove.tomography.operator_utils.ES).norm(FROBENIUS) < o_ut.EPS
    assert (states[2] - grove.tomography.operator_utils.GS).norm(FROBENIUS) < o_ut.EPS
    assert (grove.tomography.operator_utils.n_qubit_ground_state(2) - qt.tensor(
        grove.tomography.operator_utils.GS, grove.tomography.operator_utils.GS)).norm(FROBENIUS) < o_ut.EPS


def test_povm():
    pi_basis = grove.tomography.operator_utils.POVM_PI_BASIS
    confusion_rate_matrix = np.eye(2)
    povm = o_ut.make_diagonal_povm(pi_basis, confusion_rate_matrix)

    assert (povm.ops[0] - pi_basis.ops[0]).norm(FROBENIUS) < o_ut.EPS
    assert (povm.ops[1] - pi_basis.ops[1]).norm(FROBENIUS) < o_ut.EPS

    with pytest.raises(o_ut.CRMUnnormalizedError):
        confusion_rate_matrix = np.array([[.8, 0.],
                                     [.3, 1.]])
        _ = o_ut.make_diagonal_povm(pi_basis, confusion_rate_matrix)

    with pytest.raises(o_ut.CRMValueError):
        confusion_rate_matrix = np.array([[.8, -.1],
                                     [.2, 1.1]])
        _ = o_ut.make_diagonal_povm(pi_basis, confusion_rate_matrix)


def test_basis_labels():
    for num_qubits, desired_labels in [(1, ['0', '1']),
                                       (2, ['00', '01', '10', '11'])]:
        generated_labels = ut.basis_labels(num_qubits)
        assert desired_labels == generated_labels


def test_visualization():
    ax = Axes3D(figure())
    # Without axis.
    ut.state_histogram(grove.tomography.operator_utils.GS, title="test")
    # With axis.
    ut.state_histogram(grove.tomography.operator_utils.GS, ax, "test")
    assert ax.get_title() == "test"

    ptX = grove.tomography.operator_utils.PAULI_BASIS.transfer_matrix(qt.to_super(
        grove.tomography.operator_utils.QX)).toarray()
    ax = Mock()
    with patch("matplotlib.pyplot.colorbar"):
        ut.plot_pauli_transfer_matrix(ptX, ax, grove.tomography.operator_utils.PAULI_BASIS.labels, "bla")
    assert ax.imshow.called
    assert ax.set_xlabel.called
    assert ax.set_ylabel.called


def test_run_in_parallel():
    cxn = Mock(spec=QuantumComputer)
    programsXY = [[Program(I(0)), Program(X(0))],
                  [Program(I(1)), Program(X(1))]]
    nsamples = 100

    res00 = [[0, 0]]*nsamples
    res11 = [[1, 1]]*nsamples

    res01 = [[0, 1]]*nsamples
    res10 = [[1, 0]]*nsamples

    cxn.run_and_measure.side_effect = [
        res00,
        res11,
    ]
    results1 = ut.run_in_parallel(programsXY, nsamples, cxn, shuffle=False)
    assert results1.tolist() == [[[100, 0],
                                  [0, 100]],
                                 [[100, 0],
                                  [0, 100]]]

    assert cxn.run_and_measure.call_args_list == [call(Program(I(0), I(1)), [0, 1], nsamples),
                                                  call(Program(X(0), X(1)), [0, 1], nsamples)]
    cxn.run_and_measure.call_args_list = []

    with patch("grove.tomography.utils.np.random.shuffle") as shuffle:

        has_shuffled = [False]

        # flip program order on first call
        def flip_shuffle(a):
            if not has_shuffled[0]:
                a[:] = a[::-1]
                has_shuffled[0] = True

        shuffle.side_effect = flip_shuffle

        # return results corresponding to 0's programs flipped
        cxn.run_and_measure.side_effect = [
            res10,
            res01,
        ]
        results2 = ut.run_in_parallel(programsXY, nsamples, cxn, shuffle=True)
        assert results2.tolist() == [[[100, 0],
                                      [0, 100]],
                                     [[100, 0],
                                      [0, 100]]]
        assert shuffle.called
        assert cxn.run_and_measure.called
        # qubit 0's programs have flipped
        assert cxn.run_and_measure.call_args_list == [call(Program(X(0), I(1)), [0, 1], nsamples),
                                                      call(Program(I(0), X(1)), [0, 1], nsamples)]

    with pytest.raises(ValueError):
        ut.run_in_parallel([[Program(X(0))], [Program(I(0))]], nsamples, cxn, shuffle=False)