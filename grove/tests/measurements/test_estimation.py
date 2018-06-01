"""
Tests for the estimation module
"""
import pytest
from mock import Mock
import numpy as np
from scipy.stats import bernoulli
from pyquil.paulis import sX, sY, sZ, sI, PauliSum, is_zero
from pyquil.quil import Program
from pyquil.gates import RY, RX
from pyquil.api import QVMConnection
from grove.measurements.estimation import (remove_imaginary_terms,
                                           get_rotation_program,
                                           get_parity,
                                           estimate_pauli_sum,
                                           CommutationError,
                                           remove_identity,
                                           estimate_locally_commuting_operator)


def test_imaginary_removal():
    """
    remove terms with imaginary coefficients from a pauli sum
    """
    test_term = 0.25 * sX(1) * sZ(2) * sX(3) + 0.25j * sX(1) * sZ(2) * sY(3)
    test_term += -0.25j * sY(1) * sZ(2) * sX(3) + 0.25 * sY(1) * sZ(2) * sY(3)
    true_term = 0.25 * sX(1) * sZ(2) * sX(3) + 0.25 * sY(1) * sZ(2) * sY(3)
    assert remove_imaginary_terms(test_term) == true_term

    test_term = (0.25 + 1j) * sX(0) * sZ(2) + 1j * sZ(2)
    # is_identity in pyquil apparently thinks zero is identity
    assert remove_imaginary_terms(test_term) == 0.25 * sX(0) * sZ(2)

    test_term = 0.25 * sX(0) * sZ(2) + 1j * sZ(2)
    assert remove_imaginary_terms(test_term) == PauliSum([0.25 * sX(0) * sZ(2)])

    with pytest.raises(TypeError):
        remove_imaginary_terms(5)

    with pytest.raises(TypeError):
        remove_imaginary_terms(sX(0))


def test_rotation_programs():
    """
    Testing the generation of post rotations
    """
    test_term = sZ(0) * sX(20) * sI(100) * sY(5)
    # note: string comparison of programs requires gates to be in the same order
    true_rotation_program = Program().inst(
        [RX(np.pi / 2)(5), RY(-np.pi / 2)(20)])
    test_rotation_program = get_rotation_program(test_term)
    assert true_rotation_program.out() == test_rotation_program.out()


def test_get_parity():
    """
    Check if our way to compute parity is correct
    """
    single_qubit_results = [[0]] * 50 + [[1]] * 50
    single_qubit_parity_results = list(map(lambda x: -2 * x[0] + 1,
                                           single_qubit_results))

    # just making sure I constructed my test properly
    assert np.allclose(np.array([1] * 50 + [-1] * 50),
                       single_qubit_parity_results)

    test_results = get_parity([sZ(5)], single_qubit_results)
    assert np.allclose(single_qubit_parity_results, test_results[0, :])

    np.random.seed(87655678)
    brv1 = bernoulli(p=0.25)
    brv2 = bernoulli(p=0.4)
    n = 500
    two_qubit_measurements = list(zip(brv1.rvs(size=n), brv2.rvs(size=n)))
    pauli_terms = [sZ(0), sZ(1), sZ(0) * sZ(1)]
    parity_results = np.zeros((len(pauli_terms), n))
    parity_results[0, :] = [-2 * x[0] + 1 for x in two_qubit_measurements]
    parity_results[1, :] = [-2 * x[1] + 1 for x in two_qubit_measurements]
    parity_results[2, :] = [-2 * (sum(x) % 2) + 1 for x in
                            two_qubit_measurements]

    test_parity_results = get_parity(pauli_terms, two_qubit_measurements)
    assert np.allclose(test_parity_results, parity_results)


def test_estimate_pauli_sum():
    """
    Full test of the estimation procedures
    """
    quantum_resource = QVMConnection()

    # type checks
    with pytest.raises(TypeError):
        estimate_pauli_sum('5', {0: 'X', 1: 'Z'}, Program(), 1.0E-3,
                           quantum_resource)

    with pytest.raises(CommutationError):
        estimate_pauli_sum([sX(0), sY(0)], {0: 'X', 1: 'Z'}, Program(), 1.0E-3,
                           quantum_resource)

    with pytest.raises(TypeError):
        estimate_pauli_sum(sX(0), {0: 'X', 1: 'Z'}, Program(), 1.0E-3,
                           quantum_resource)

    # mock out qvm
    np.random.seed(87655678)
    brv1 = bernoulli(p=0.25)
    brv2 = bernoulli(p=0.4)
    n = 500
    two_qubit_measurements = list(zip(brv1.rvs(size=n), brv2.rvs(size=n)))
    pauli_terms = [sZ(0), sZ(1), sZ(0) * sZ(1)]

    fakeQVM = Mock(spec=QVMConnection())
    fakeQVM.run = Mock(return_value=two_qubit_measurements)
    mean, cov, estimator_var, shots = estimate_pauli_sum(pauli_terms,
                                                         {0: 'Z', 1: 'Z'},
                                                         Program(),
                                                         1.0E-1, fakeQVM)
    parity_results = np.zeros((len(pauli_terms), n))
    parity_results[0, :] = [-2 * x[0] + 1 for x in two_qubit_measurements]
    parity_results[1, :] = [-2 * x[1] + 1 for x in two_qubit_measurements]
    parity_results[2, :] = [-2 * (sum(x) % 2) + 1 for x in
                            two_qubit_measurements]

    assert np.allclose(np.cov(parity_results), cov)
    assert np.isclose(np.sum(np.mean(parity_results, axis=1)), mean)
    assert np.isclose(shots, n)
    variance_to_beat = np.sum(cov) / n
    assert np.isclose(variance_to_beat, estimator_var)

    # Double the shots by ever so slightly decreasing variance bound
    double_two_q_measurements = two_qubit_measurements + two_qubit_measurements
    mean, cov, estimator_var, shots = estimate_pauli_sum(pauli_terms,
                                                         {0: 'Z', 1: 'Z'},
                                                         Program(),
                                                         variance_to_beat - \
                                                         1.0E-8, fakeQVM)

    parity_results = np.zeros((len(pauli_terms), 2 * n))
    parity_results[0, :] = [-2 * x[0] + 1 for x in double_two_q_measurements]
    parity_results[1, :] = [-2 * x[1] + 1 for x in double_two_q_measurements]
    parity_results[2, :] = [-2 * (sum(x) % 2) + 1 for x in
                            double_two_q_measurements]

    assert np.allclose(np.cov(parity_results), cov)
    assert np.isclose(np.sum(np.mean(parity_results, axis=1)), mean)
    assert np.isclose(shots, 2 * n)
    assert np.isclose(np.sum(cov) / (2 * n), estimator_var)


def test_identity_removal():
    test_term = 0.25 * sX(1) * sZ(2) * sX(3) + 0.25j * sX(1) * sZ(2) * sY(3)
    test_term += -0.25j * sY(1) * sZ(2) * sX(3) + 0.25 * sY(1) * sZ(2) * sY(3)
    identity_term = 200 * sI(5)

    new_psum, identity_term_result = remove_identity(identity_term + test_term)
    assert test_term == new_psum
    assert identity_term_result == identity_term
