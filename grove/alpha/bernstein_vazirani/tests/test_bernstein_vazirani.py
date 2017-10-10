import numpy as np
import pyquil.api as api
import pyquil.quil as pq
import pytest
from pyquil.gates import X
from mock import patch

from pyquil.api import SyncConnection
from pyquil.quil import Program

from grove.alpha.bernstein_vazirani.bernstein_vazirani import bernstein_vazirani, oracle_function, run_bernstein_vazirani

def _create_expected_prog():
    expected_prog = Program()
    expected_prog.inst("X 3")
    expected_prog.inst("H 3")
    expected_prog.inst("H 0")
    expected_prog.inst("H 1")
    expected_prog.inst("H 2")
    expected_prog.inst("X 3")
    expected_prog.inst("CNOT 2 3")
    expected_prog.inst("CNOT 1 3")
    expected_prog.inst("H 0")
    expected_prog.inst("H 1")
    expected_prog.inst("H 2")
    return expected_prog

def test_bernstein_vazirani():
    orc = oracle_function([1, 1, 0], 1, [0, 1, 2], 3)
    bv_prog = bernstein_vazirani(orc, [0, 1, 2], 3)

    with patch("pyquil.api.SyncConnection") as qvm:
        # Need to mock multiple returns as an iterable
        qvm.run_and_measure.side_effect = [
            ([0, 1, 1], ),
            ([1], )
        ]


    # qvm = SyncConnection("http://127.0.0.1:5000")

    bv_a, bv_b, bv_program = run_bernstein_vazirani(qvm, orc, [0, 1, 2], 3)

    assert bv_a == [1, 1, 0]
    assert bv_b == 1
    assert bv_program.__str__() == _create_expected_prog().__str__()


@pytest.mark.skip(reason="Must add support for Forest connections in testing")
class TestOracleFunction(object):
    def test_one_qubit(self):
        vec_a = np.array([1])
        b = 0
        for x in range(2 ** len(vec_a)):
            _oracle_test_helper(vec_a, b, x)

    def test_two_qubits(self):
        vec_a = np.array([1, 0])
        b = 1
        for x in range(2 ** len(vec_a)):
            _oracle_test_helper(vec_a, b, x)

    def test_three_qubits(self):
        vec_a = np.array([0, 0, 0])
        b = 0
        for x in range(2 ** len(vec_a)):
            _oracle_test_helper(vec_a, b, x)

    def test_four_qubits(self):
        vec_a = np.array([1, 1, 1, 1])
        b = 1
        for x in range(2 ** len(vec_a)):
            _oracle_test_helper(vec_a, b, x)


@pytest.mark.skip(reason="Must add support for Forest connections in testing")
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

    for i in range(len(bitstring)):
        if bitstring[-1 - i] == '1':
            p.inst(X(i))

    oracle = oracle_function(vec_a, b, qubits, ancilla)
    p += oracle
    results = cxn.run_and_measure(p, [ancilla], trials)
    a_dot_x = np.binary_repr(int(''.join(list(map(str, vec_a))), 2) & x).count("1")
    expected = (a_dot_x + b) % 2
    for result in results:
        y = result[0]
        assert y == expected
