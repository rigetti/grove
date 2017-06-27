import pyquil.api as api
import numpy as np
import pyquil.quil as pq
from pyquil.gates import X, I

from grove.bernstein_vazirani.bernstein_vazirani import bernstein_vazirani, oracle_function
import pytest

#@pytest.mark.skip(reason="Must add support for Forest connections in testing")
class TestOracleFunction(object):
    def test_one_qubit(self):
        vec_a = np.array([1])
        b = 0
        for x in range(2**len(vec_a)):
            _oracle_test_helper(vec_a, b, x)

    def test_two_qubits(self):
        vec_a = np.array([1, 0])
        b = 1
        for x in range(2**len(vec_a)):
            _oracle_test_helper(vec_a, b, x)

    def test_three_qubits(self):
        vec_a = np.array([0, 0, 0])
        b = 0
        for x in range(2**len(vec_a)):
            _oracle_test_helper(vec_a, b, x)

    def test_four_qubits(self):
        vec_a = np.array([1, 1, 1, 1])
        b = 1
        for x in range(2**len(vec_a)):
            _oracle_test_helper(vec_a, b, x)

#@pytest.mark.skip(reason="Must add support for Forest connections in testing")
class TestBernsteinVazirani(object):
    def test_one_qubit_all_zeros(self):
        _bv_test_helper(np.array([0]), 0)

    def test_two_qubit_all_ones(self):
        _bv_test_helper(np.array([1, 1]), 1)

    def test_four_qubit_symmetric(self):
        _bv_test_helper(np.array([1, 0, 0, 1]), 1)

    def test_five_qubits_asymmetric(self):
        _bv_test_helper(np.array([0, 0, 1, 0, 1]), 0)


def _bv_test_helper(vec_a, b, trials=1):
    qubits = range(len(vec_a))
    ancilla = len(vec_a)
    oracle = oracle_function(vec_a, b, qubits, ancilla)
    bv_program = bernstein_vazirani(oracle, qubits, ancilla)
    cxn = api.SyncConnection()
    results = cxn.run_and_measure(bv_program, qubits, trials)
    for result in results:
        bv_a = result[::-1]
        assert bv_a == list(vec_a)


def _oracle_test_helper(vec_a, b, x, trials=1):
    qubits = range(len(vec_a))
    ancilla = len(vec_a)

    cxn = api.SyncConnection()
    bitstring = np.binary_repr(x, len(qubits))
    p = pq.Program()

    # TODO get rid of I's when strange X behavior is understood
    for i in range(len(bitstring)):
        p.inst(I(i))
        if bitstring[-1-i] == '1':
            p.inst(X(i))

    p.inst(I(ancilla))
    oracle = oracle_function(vec_a, b, qubits, ancilla)
    p += oracle
    results = cxn.run_and_measure(p, [ancilla], trials)
    expected = (np.binary_repr(int(''.join(map(str, vec_a)), 2) & x).count("1") + b) % 2
    for result in results:
        y = result[0]
        assert y == expected

