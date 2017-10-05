##############################################################################
# Copyright 2016-2017 Rigetti Computing
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
import pyquil.quil as pq
import pytest
from pyquil.gates import *

from grove.alpha.amplification.amplification import amplify, n_qubit_control, diffusion_operator
from grove.pyquil_utilities import prog_equality

# Normal operation

# Setup some variables to reuse
A = pq.Program().inst(H(0)).inst(H(1)).inst(H(2))
cz_gate = n_qubit_control([1], 2, np.array([[1, 0], [0, -1]]), "CZ")
oracle = pq.Program().inst()
qubits = [0, 1, 2]
iters = 2


def test_qubit_control():
    """
    Tests the n_qubit_control on a generic number of qubits
    """

    # Creates a controlled Z gate from index 0 to index 1
    created = n_qubit_control([0], 1, np.array([[1, 0], [0, -1]]), "CZ")
    assert np.array_equal(np.array(created.defined_gates[0].matrix),
                          np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                                    [0, 0, 0, -1]]))


def test_diffusion_operator():
    """
    Checks that the diffusion operator outputs the correct operation
    """

    created = diffusion_operator([0, 1])
    desired = pq.Program()
    desired.inst(X(0))
    desired.inst(X(1))
    desired.inst(H(1))
    desired.inst(RZ(-3.141592653589793, 0))
    desired.inst(CNOT(0, 1))
    desired.inst(RZ(-3.141592653589793, 0))
    desired.inst(H(1))
    desired.inst(X(0))
    desired.inst(X(1))

    assert desired.out() == created.out()


def test_amplify():
    """
    Test the generic usage of amplify
    """

    # Essentially Grover's to select 011 or 111
    desired = A + cz_gate + A.dagger() + diffusion_operator(
        qubits) + A + cz_gate + A.dagger() + diffusion_operator(qubits) + A
    created = amplify(A, cz_gate, qubits, iters)

    prog_equality(desired, created)


def test_amplify_init():
    """
    Test the usage of amplify without init
    """
    # Essentially Grover's to select 011 or 111
    desired = cz_gate + A.dagger() + diffusion_operator(
        qubits) + A + cz_gate + A.dagger() + diffusion_operator(qubits) + A
    created = amplify(A, cz_gate, qubits, iters)

    prog_equality(desired, created)


# Edge Cases

def test_edge_case_amplify_0_iters():
    """
    Checks that the number of iterations needed to be greater than 0
    """
    with pytest.raises(ValueError):
        amplify(A, oracle, qubits, 0)


def test_edge_case_A_none():
    """
    Checks that A cannot be None
    """
    with pytest.raises(ValueError):
        amplify(None, oracle, qubits, iters)


def test_edge_case_oracle_none():
    """
    Checks that U_w cannot be None
    """
    with pytest.raises(ValueError):
        amplify(A, None, qubits, iters)


def test_edge_case_qubits_empty():
    """
    Checks that the list of qubits to apply the grover
    diffusion operator to must be non-empty
    """
    with pytest.raises(ValueError):
        amplify(A, oracle, [], iters)


def test_diffusion_operator_empty():
    """
    Checks that the list of qubits to apply the grover
    diffusion operator to must be non-empty
    """
    with pytest.raises(ValueError):
        diffusion_operator([])


def test_n_qubit_control_unitary_none():
    """
    Checks that the n qubit control object needs a
    unitary as a numpy matrix
    """
    with pytest.raises(ValueError):
        n_qubit_control([0, 1, 2], 3, "not an array", "BAD")


def test_n_qubit_control_target_none():
    """
    Checks that the n qubit control object needs a
    list of control qubits
    """
    with pytest.raises(ValueError):
        n_qubit_control([0, 1, 2], -1, np.array([[1, 0], [0, 1]]), "IDENT")


def test_n_qubit_control_name_bad():
    """
    Checks that the n qubit control object needs a
    list of control qubits
    """
    with pytest.raises(ValueError):
        n_qubit_control([0, 1, 2], 4, np.array([[1, 0], [0, 1]]), "")
