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
from pyquil.gates import X, H, RZ, CZ

from grove.amplification.amplification import amplification_circuit, diffusion_program
from grove.tests.utils.utils_for_testing import prog_equality

triple_hadamard = pq.Program().inst(H(0)).inst(H(1)).inst(H(2))
cz_gate = pq.Program(CZ(0, 1))
oracle = pq.Program().inst()
qubits = [0, 1, 2]
iters = 2


def test_diffusion_operator():
    """
    Checks that the diffusion operator outputs the correct operation
    """
    created = diffusion_program([0, 1])
    desired = pq.Program()
    for def_gate in created.defined_gates:
        desired.defgate(def_gate.name, def_gate.matrix)
    desired.inst(X(0))
    desired.inst(X(1))
    desired.inst(H(1))
    desired.inst(RZ(-np.pi, 0))
    # There should be only one defined gate -- the CNOT.
    desired.inst((created.defined_gates[0].name, 0, 1))
    desired.inst(RZ(-np.pi, 0))
    desired.inst(H(1))
    desired.inst(X(0))
    desired.inst(X(1))
    assert prog_equality(created, desired)


def test_amplify():
    """
    Test the generic usage of amplify
    """

    # Essentially Grover's to select 011 or 111
    desired = triple_hadamard + cz_gate + triple_hadamard.dagger() + diffusion_program(
        qubits) + triple_hadamard + cz_gate + triple_hadamard.dagger() + diffusion_program(qubits) + triple_hadamard
    created = amplification_circuit(triple_hadamard, cz_gate, qubits, iters)

    prog_equality(desired, created)


def test_amplify_init():
    """
    Test the usage of amplify without init
    """
    # Essentially Grover's to select 011 or 111
    desired = cz_gate + triple_hadamard.dagger() + diffusion_program(
        qubits) + triple_hadamard + cz_gate + triple_hadamard.dagger() + diffusion_program(qubits) + triple_hadamard
    created = amplification_circuit(triple_hadamard, cz_gate, qubits, iters)

    prog_equality(desired, created)
