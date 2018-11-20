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
from pyquil import Program
from pyquil.gates import X, H, Z, RZ, CZ, CNOT

from grove.amplification.amplification import (amplification_circuit,
                                               decomposed_diffusion_program,
                                               diffusion_program)


triple_hadamard = Program().inst(H(2)).inst(H(1)).inst(H(0))
cz_gate = Program(CZ(0, 1))
oracle = Program().inst()
qubits = [0, 1, 2]
iters = 2


def test_diffusion_operator():
    """
    Checks that the diffusion operator outputs the correct operation
    """
    created = decomposed_diffusion_program(qubits[:2])
    desired = Program()
    for def_gate in created.defined_gates:
        desired.defgate(def_gate.name, def_gate.matrix)
    qubit0 = qubits[0]
    qubit1 = qubits[1]
    desired.inst(X(qubit0))
    desired.inst(X(qubit1))
    desired.inst(H(qubit1))
    desired.inst(RZ(-np.pi, qubit0))
    desired.inst(CNOT(qubit0, qubit1))
    desired.inst(RZ(-np.pi, qubit0))
    desired.inst(H(qubit1))
    desired.inst(X(qubit0))
    desired.inst(X(qubit1))
    assert desired == created


def test_amplify():
    """
    Test the generic usage of amplify
    """
    # Essentially Grover's to select 011 or 111
    desired = (triple_hadamard.dagger()
               + cz_gate
               + triple_hadamard.dagger()
               + diffusion_program(qubits)
               + triple_hadamard
               + cz_gate
               + triple_hadamard.dagger()
               # We take care to only add the instructions, and not redefine the gate.
               + diffusion_program(qubits).instructions
               + triple_hadamard)
    created = amplification_circuit(triple_hadamard, cz_gate, qubits, iters)
    assert desired == created


def test_trivial_diffusion():
    qubits = [0]
    created = decomposed_diffusion_program(qubits)
    desired = Program().inst(Z(qubits[0]))
    assert created == desired
